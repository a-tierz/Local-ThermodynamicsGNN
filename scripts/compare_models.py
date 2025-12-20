import os
import json
import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import time
from glob import glob
from torch_geometric.loader import DataLoader
from src.gnn_nodal import NodalGNN
from src.gnn import GNN
from src.dataLoader.dataset import GraphDataset

MODEL_CLASSES = {
    'GNN': GNN,
    'NodalGNN': NodalGNN,
}

def load_model(weights_path, config_path, device):
    """Load a model from checkpoint and configuration."""
    with open(config_path, 'r') as f:
        dInfo = json.load(f)
    
    # Try to find scaler
    ckpt_dir = os.path.dirname(weights_path)
    potential_scaler = os.path.join(ckpt_dir, 'scaler.pkl')
    
    if os.path.exists(potential_scaler):
        with open(potential_scaler, 'rb') as f:
            scaler = pickle.load(f)
    else:
        # If not in weights dir, check dataset path in config
        train_dset_path = os.path.join('data', 'datasets', dInfo['dataset']['datasetPaths']['train'])
        scaler_dset_path = train_dset_path.replace('.pt', '.scaler.pkl')
        if os.path.exists(scaler_dset_path):
            with open(scaler_dset_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f"No scaler found for {weights_path}")

    # Determine dimensions (assuming they can be inferred or are in config)
    # We create a dummy dataset or use config to get dims
    # Extracting dims logic from GraphDataset.dims
    z_dim = len(dInfo['dataset']['state_variables'])
    dims = {
        'z': z_dim,
        'q': dInfo['dataset']['q_dim'],
        'q_0': dInfo['dataset']['q0_dim'],
        'n': 1,
        'f': dInfo['dataset']['external_force_dim'],
        'g': dInfo['dataset']['g_dim'],
    }

    model_type = dInfo['model'].get('type', 'NodalGNN')
    model_class = MODEL_CLASSES.get(model_type, NodalGNN)
    
    model = model_class.load_from_checkpoint(
        weights_path, 
        dt_info=dInfo, 
        dims=dims, 
        scaler=scaler, 
        save_folder=''
    )
    model.to(device)
    model.eval()
    return model, dInfo, scaler

def compute_one_step_metrics(model, dataloader, device, state_variables):
    """Compute MSE and MAE for one-step prediction."""
    all_errors = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z1_pred, z1_gt, _ = model.predict_step(batch, 0)
            
            # Use only fluid particles if applicable
            mask = batch.n == 1
            e = (z1_pred[mask] - z1_gt[mask]).cpu().numpy()
            all_errors.append(e)
            
    all_errors = np.concatenate(all_errors, axis=0)
    mse = np.mean(all_errors**2, axis=0)
    mae = np.mean(np.abs(all_errors), axis=0)
    
    inference_time = (time.time() - start_time) / len(dataloader)
    
    metrics = {}
    for i, var in enumerate(state_variables):
        metrics[f"MSE_{var}"] = mse[i]
        metrics[f"MAE_{var}"] = mae[i]
    metrics["avg_inference_time"] = inference_time
    
    return metrics

def run_rollout_simulation(model, simulation_data, device, dInfo, threshold_mult=10.0):
    """Run rollout and monitor for divergence."""
    num_steps = len(simulation_data)
    state_variables = dInfo['dataset']['state_variables']
    radius = dInfo['dataset']['radius_connectivity']
    
    # Get ground truth max for thresholding
    all_y = torch.cat([d.y for d in simulation_data], dim=0)
    gt_max = torch.abs(all_y).max(dim=0).values.cpu().numpy()
    thresholds = gt_max * threshold_mult
    
    z_net = []
    z_gt = []
    
    # Initial state
    current_snap = simulation_data[0].clone().to(device)
    z_net.append(current_snap.x[current_snap.n == 1].cpu())
    z_gt.append(current_snap.x[current_snap.n == 1].cpu())
    
    steps_until_divergence = num_steps
    diverged = False
    
    for t in range(num_steps):
        try:
            with torch.no_grad():
                z_next_denorm, z_gt_t1, _ = model.predict_step(current_snap, t)
            
            # Check for NaN
            if torch.isnan(z_next_denorm).any():
                print(f"Rollout diverged at step {t}: NaN detected")
                steps_until_divergence = t
                diverged = True
                break
                
            # Check for threshold violation
            mask = current_snap.n == 1
            pred_vals = z_next_denorm[mask].cpu().numpy()
            if (np.abs(pred_vals) > thresholds).any():
                print(f"Rollout diverged at step {t}: Threshold exceeded")
                steps_until_divergence = t
                diverged = True
                break
            
            z_net.append(z_next_denorm[mask].cpu())
            z_gt.append(z_gt_t1[mask].cpu())
            
            # Prepare next step
            if t < num_steps - 1:
                next_snap = simulation_data[t+1].clone()
                next_snap.x = z_next_denorm # Update state for next step
                # Note: In fluid simulations, connectivity might need recalculation
                # But here we follow evaluate.py logic
                current_snap = next_snap.to(device)
                
        except Exception as e:
            print(f"Rollout failed at step {t}: {e}")
            steps_until_divergence = t
            diverged = True
            break
            
    # Calculate error only for stable portion
    if len(z_net) > 1:
        z_net_stack = torch.stack(z_net[1:])
        z_gt_stack = torch.stack(z_gt[1:])
        error = (z_net_stack - z_gt_stack).numpy()
        rmse_rollout = np.sqrt(np.mean(error**2, axis=(0, 1)))
    else:
        rmse_rollout = np.array([np.nan] * len(state_variables))
        
    return {
        "steps_until_divergence": steps_until_divergence,
        "diverged": diverged,
        "rmse_rollout": rmse_rollout,
        "total_steps": num_steps
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', ngettext='-m', action='append', nargs=2, metavar=('WEIGHTS', 'CONFIG'), 
                        help='Pairs of weights and config files')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test .pt files')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='comparison_results.md')
    args = parser.parse_args()

    device = torch.device(args.device)
    test_files = glob(os.path.join(args.test_dir, "*.pt"))
    test_files.sort()
    
    print(f"Found {len(test_files)} test simulations.")
    
    results = []
    
    for weights_path, config_path in args.models:
        model_name = os.path.basename(weights_path)
        print(f"\nEvaluating model: {model_name}")
        
        model, dInfo, scaler = load_model(weights_path, config_path, device)
        state_vars = dInfo['dataset']['state_variables']
        
        # 1. One-step evaluation
        # Load all test data for one-step
        all_test_data = []
        for f in test_files:
            all_test_data.extend(torch.load(f, weights_only=False))
        
        dataloader = DataLoader(all_test_data, batch_size=1)
        one_step_metrics = compute_one_step_metrics(model, dataloader, device, state_vars)
        
        # 2. Rollout evaluation
        rollout_results = []
        for f in test_files:
            sim_data = torch.load(f, weights_only=False)
            res = run_rollout_simulation(model, sim_data, device, dInfo)
            rollout_results.append(res)
            
        # Aggregate rollout metrics
        avg_sud = np.mean([r['steps_until_divergence'] for r in rollout_results])
        pct_diverged = np.mean([1 if r['diverged'] else 0 for r in rollout_results]) * 100
        
        # Average RMSE for stable portions (avoiding NaNs)
        rollout_rmses = [r['rmse_rollout'] for r in rollout_results if not np.isnan(r['rmse_rollout']).any()]
        if rollout_rmses:
            avg_rollout_rmse = np.mean(rollout_rmses, axis=0)
        else:
            avg_rollout_rmse = np.array([np.nan] * len(state_vars))
            
        model_results = {
            "name": model_name,
            "one_step": one_step_metrics,
            "rollout": {
                "avg_sud": avg_sud,
                "pct_diverged": pct_diverged,
                "avg_rmse": avg_rollout_rmse
            }
        }
        results.append(model_results)

    # Generate Markdown Table
    with open(args.output, 'w') as f:
        f.write("# Model Comparison Results\n\n")
        
        # Table Header
        headers = ["Model", "SUD (Avg)", "Diverged (%)", "Inference Time (s)"]
        state_vars = results[0]['one_step'].keys()
        # Filter only MSE
        mse_vars = [k for k in state_vars if k.startswith("MSE_")]
        headers += [f"{v} (One-step)" for v in mse_vars]
        headers += [f"{v.replace('MSE_', 'RMSE_')} (Rollout)" for v in mse_vars]
        
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        
        for res in results:
            row = [
                res['name'],
                f"{res['rollout']['avg_sud']:.1f}",
                f"{res['rollout']['pct_diverged']:.1f}%",
                f"{res['one_step']['avg_inference_time']:.4f}"
            ]
            for v in mse_vars:
                row.append(f"{res['one_step'][v]:.2e}")
            
            for i in range(len(mse_vars)):
                val = res['rollout']['avg_rmse'][i]
                row.append(f"{val:.2e}" if not np.isnan(val) else "N/A")
                
            f.write("| " + " | ".join(row) + " |\n")

    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
