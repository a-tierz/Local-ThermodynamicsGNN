import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pickle
import argparse
import time
import datetime
from glob import glob
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from torch_geometric.loader import DataLoader
from src.gnn_nodal import NodalGNN
from src.gnn import GNN
from src.dataLoader.dataset import GraphDataset
from src.evaluate import compute_error
from src.utils.utils import generate_folder, set_seed
from src.utils.plots import plotError, plot_2D, plot_3D, plot_2D_image
from torch_geometric.nn import radius_graph

MODEL_CLASSES = {
    'GNN': GNN,
    'NodalGNN': NodalGNN,
}

def load_model(weights_path, config_path, device):
    """Load a model from checkpoint and configuration."""
    with open(config_path, 'r') as f:
        dInfo = json.load(f)

    # Set seed for reproducibility
    seed = dInfo['model'].get('seed', 42)
    set_seed(seed)
    
    ckpt_dir = os.path.dirname(weights_path)
    potential_scaler = os.path.join(ckpt_dir, 'scaler.pkl')
    
    if os.path.exists(potential_scaler):
        with open(potential_scaler, 'rb') as f:
            scaler = pickle.load(f)
    else:
        train_dset_path = os.path.join('data', 'datasets', dInfo['dataset']['datasetPaths']['train'])
        scaler_dset_path = train_dset_path.replace('.pt', '.scaler.pkl')
        if os.path.exists(scaler_dset_path):
            with open(scaler_dset_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            print("Warning: No scaler found. Recalculating from dataset info (this might be inaccurate).")
            # This is a fallback, ideally the scaler should be provided
            scaler = None 

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

def run_rollout(model, simulation_data, device, dInfo, threshold_mult=10.0):
    """Run rollout and monitor for divergence."""
    num_steps = len(simulation_data)
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
    
    diverged = False
    step_diverged = num_steps
    
    for t in range(num_steps):
        try:
            with torch.no_grad():
                z_next_denorm, z_gt_t1, _ = model.predict_step(current_snap, t)
            
            # Check for NaN
            if torch.isnan(z_next_denorm).any():
                diverged = True
                step_diverged = t
                break
                
            # Check for threshold violation
            mask = current_snap.n == 1
            pred_vals = z_next_denorm[mask].cpu().numpy()
            if (np.abs(pred_vals) > thresholds).any():
                diverged = True
                step_diverged = t
                break
            
            z_net.append(z_next_denorm[mask].cpu())
            z_gt.append(z_gt_t1[mask].cpu())
            # Prepare next step
            if t < num_steps - 1:
                next_snap = simulation_data[t+1].clone()
                next_snap.x = z_next_denorm # Update state for next step
                
                # Update connectivity if fluid
                if dInfo['dataset']['type'] == 'fluid':
                    pos = z_next_denorm[:, :3].clone()
                    next_snap.edge_index = radius_graph(pos, r=radius, loop=False, 
                                                       flow='source_to_target', max_num_neighbors=1000)
                
                current_snap = next_snap.to(device)
                
        except Exception as e:
            print(f"Error during rollout at step {t}: {e}")
            diverged = True
            step_diverged = t
            break
            
    # Return as tensors [steps, nodes, variables]
    return torch.stack(z_net), torch.stack(z_gt), step_diverged, diverged

def main():
    parser = argparse.ArgumentParser(description='Evaluate a single GNN model')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt weights')
    parser.add_argument('--config', type=str, required=True, help='Path to .json config')
    parser.add_argument('--test_dir', type=str, required=True, help='Directory with test .pt files')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--plot_sim_idx', type=int, default=0, help='Index of simulation to plot')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = os.path.join(args.output_dir, os.path.basename(args.weights).replace('.ckpt', '') + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {args.weights}...")
    model, dInfo, scaler = load_model(args.weights, args.config, device)
    state_vars = dInfo['dataset']['state_variables']
    dataset_dim = dInfo['dataset']['dataset_dim']

    test_files = glob(os.path.join(args.test_dir, "*.pt"))
    test_files.sort()
    print(f"Found {len(test_files)} test simulations.")

    all_metrics = {
        "one_step": {},
        "rollout": []
    }

    # 1. One-step Evaluation (all samples)
    print("Computing one-step metrics...")
    all_errors = []
    total_samples = 0
    with torch.no_grad():
        for f in test_files:
            sim_data = torch.load(f, weights_only=False)
            dataloader = DataLoader(sim_data, batch_size=1)
            for batch in dataloader:
                batch = batch.to(device)
                z1_pred, z1_gt, _ = model.predict_step(batch, 0)
                mask = batch.n == 1
                e = (z1_pred[mask] - z1_gt[mask]).cpu().numpy()
                all_errors.append(e)
                total_samples += 1

    all_errors = np.concatenate(all_errors, axis=0)
    mse = np.mean(all_errors**2, axis=0)
    mae = np.mean(np.abs(all_errors), axis=0)
    
    for i, var in enumerate(state_vars):
        all_metrics["one_step"][f"MSE_{var}"] = float(mse[i])
        all_metrics["one_step"][f"MAE_{var}"] = float(mae[i])

    # 2. Rollout Evaluation
    print("Running rollout simulations...")
    for i, f in enumerate(test_files):
        print(f"  Simulation {i}: {os.path.basename(f)}")
        sim_data = torch.load(f, weights_only=False)
        z_net, z_gt, sud, diverged = run_rollout(model, sim_data, device, dInfo)
        
        sim_res = {
            "file": os.path.basename(f),
            "sud": int(sud),
            "total_steps": len(sim_data),
            "diverged": bool(diverged)
        }
        
        # Error metrics for the stable part
        if len(z_net) > 1:
            error, L2_list = compute_error(z_net[1:], z_gt[1:], state_vars)
            sim_res["rmse"] = {k: float(v) for k, v in error.items()}
            
            # Visualization for the requested simulation
            if i == args.plot_sim_idx:
                print(f"  Generating plots for simulation {i}...")
                plotError(z_gt, z_net, L2_list, state_vars, dataset_dim, output_dir)
                
                gif_path = os.path.join(output_dir, f"rollout_sim_{i}.gif")
                if dataset_dim == '2D':
                    plot_2D(z_net.numpy(), z_gt.numpy(), gif_path, var=4 if len(state_vars) > 4 else 0)
                else:
                    plot_3D(z_net.numpy(), z_gt.numpy(), gif_path, var=-1)
                    
        all_metrics["rollout"].append(sim_res)

    # Summary
    avg_sud = np.mean([r['sud'] for r in all_metrics['rollout']])
    pct_diverged = np.mean([1 if r['diverged'] else 0 for r in all_metrics['rollout']]) * 100
    
    print("\n--- Evaluation Summary ---")
    print(f"One-step MSE (avg): {np.mean(mse):.2e}")
    print(f"Avg Steps Until Divergence (SUD): {avg_sud:.1f} / {len(sim_data)}")
    print(f"Percent Diverged: {pct_diverged:.1f}%")
    
    # Save metrics JSON
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)
        
    print(f"\nDetailed results and plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
