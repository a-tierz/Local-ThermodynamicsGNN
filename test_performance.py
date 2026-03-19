import pytorch_lightning as pl
from pathlib import Path
import torch
import argparse
import os
from types import SimpleNamespace
import json
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
# from amb.metrics import se_inf

import numpy as np
from src.dataLoader.dataset import GraphDataset
from src.gnn_nodal import NodalGNN
# from src.gnn_global import NodalGNN
# from src.gnn import NodalGNN
# from src.spnn import NodalGNN
from src.utils.utils import str2bool
from src.evaluate import generate_results, roll_out


pl.seed_everything(42, workers=True)

noise_std = 0.
device = 'mps'  # test example 10 with 750 snapshot takes mps = 81 sec || cpu  = 125 sec || gpu = ?? sec


def se_inf(data_ground_truth, data_predicted):
    """
    Calcula el Error Relativo normalizado por el Máximo (Norma Infinito).
    Retorna el ratio adimensional (ej: 0.05 significa 5% de error respecto al pico).
    
    Formula: RMSE_snapshot / Max_Abs_GT_snapshot
    """
    x, y = data_ground_truth, data_predicted

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)
        y = np.asarray(y)

    # Asegurar dimensiones [Snapshots, Nodos, Vars]
    if len(x.shape) == 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

    # 1. Numerador: RMSE (Root Mean Square Error) por snapshot
    # Nos dice el error "promedio" de un nodo típico, no la suma total.
    diff = x - y
    # Mean sobre nodos (axis=1) y vars si están aplanadas
    mse_per_step = np.mean(diff ** 2, axis=1) 
    rmse_per_step = np.sqrt(mse_per_step) # [Snapshots, Vars]

    # 2. Denominador: Norma Infinito (Valor Máximo Absoluto) del Ground Truth
    # El valor más grande que ocurre en la malla en ese instante.
    max_val_per_step = np.max(np.abs(x), axis=1) # [Snapshots, Vars]

    # Evitar división por cero
    epsilon = 1e-8
    
    # 3. Ratio: (Error Promedio) / (Señal Máxima)
    # Esto devuelve un valor adimensional lineal (no al cuadrado)
    # Si quieres el cuadrado (MSE relativo), eleva todo al final, 
    # pero para interpretar es mejor lineal (0.1 = 10% de error).
    relative_error = rmse_per_step / (max_val_per_step + epsilon)

    # Si tu código espera una lista plana o media, ajusta aquí.
    # Como tu código hace 'mean' luego, devolvemos el array completo.
    return relative_error

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MeshGraph Simulation')
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset_dir', type=str, default='data/datasets/test_V66', help='Directory containing dataset')
    parser.add_argument('--dset_name', default=r'dataset_Water3D.json', type=str, help='dataset directory')
    parser.add_argument('--output_dir_exp', default=r'outputs/experiments/foam/3D/', type=str,
                    help='output directory')
    parser.add_argument('--split', type=str, default="test")
    parser.add_argument('--steps', type=int, default=60)

    args = parser.parse_args()
    # path = Path(args.path)
    dataset_dir = args.dataset_dir
    steps = args.steps
    split = args.split

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    f = open(os.path.join('configs', args.dset_name))
    dInfo = json.load(f)
    dim_data = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3

    train_set = GraphDataset(dInfo, os.path.join(args.dset_dir, 'datasets', dInfo['dataset']['datasetPaths']['train']))
    train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'])


    scaler = train_set.get_stats()


    for name_treining in os.listdir(r'data/weights_test'):
        output_dir_exp = 'outputs/cosa'
        # Instantiate model
        nodal_gnn = NodalGNN(train_set.dims, scaler, dInfo, output_dir_exp)
        nodal_gnn.to(device)

        load_name = name_treining
        # load_name = args.pretrain_weights
        load_path = os.path.join(args.dset_dir, 'weights_test', load_name)
        checkpoint = torch.load(load_path, map_location='cuda')

        dif = False
        if dif:
            old_state_dict = checkpoint['state_dict']
            new_state_dict = nodal_gnn.state_dict()
            updated_state_dict = {k: v for k, v in old_state_dict.items() if
                                  k in new_state_dict and v.shape == new_state_dict[k].shape}

            new_state_dict.update(updated_state_dict)
        else:
            new_state_dict = checkpoint['state_dict']
        nodal_gnn.load_state_dict(new_state_dict)

        nodal_gnn.eval()
        trainer = pl.Trainer(accelerator="cpu",
                             profiler="simple")
    ###

        # with open(path.parent.parent / 'config.json', 'r') as file:
        #     config = json.load(file)

        # args = SimpleNamespace(**config)
        # args.path = path
        # radius = args.radius
        # mp_steps = args.mp_steps
        # layers = args.layers
        # hidden = args.hidden
        # shared_mp = args.shared_mp
        # output_size = 10 if args.velocity else 4



        mse_pos, mse_velocity, mse_energy, global_se_pos, global_se_velocity, global_se_energy = [], [], [], [], [], []
        mse_inf_pos, mse_inf_velocity,mse_inf_energy, global_se_inf_pos, global_se_inf_velocity , global_se_inf_energy = [], [], [], [], [], []
        mse_pos_dict, mse_velocity_dict, mse_energy_dict = {}, {}, {}
        mse_inf_pos_dict, mse_inf_velocity_dict, mse_inf_energy_dict = {}, {}, {}

        path_save = 'outputs/rollouts/test_'+ name_treining[:-4]

        Path(path_save).mkdir(exist_ok=True, parents=True)
        path_save = str(path_save)

        trajectory_dir = dataset_dir

        for tra_name in os.listdir(trajectory_dir):
            print(tra_name)
            test_set = GraphDataset(dInfo,
                                    os.path.join(trajectory_dir, tra_name))

            test_dataloader = DataLoader(test_set, batch_size=1, )  # dInfo['model']['batch_size'])

            predictions, targets, t = roll_out(nodal_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'], dInfo['dataset']['type'])

            predictions = predictions[1:steps,:,:]
            targets = targets[1:steps,:,:]
            # tra = tra_name.stem
            # print(f'Solving trajectory {tra}')
            # rollout = RolloutCallback(dataset_dir=dataset_dir, split=split, trajectory=tra, all_velocity=args.velocity)
            # results, n, edges, faces = simulator.rollout(simulator, rollout.loader, max_steps=steps)
            # predictions, targets = results[0], results[1]

            # make_pyvista_video(predictions, targets, faces, var_index=-1, max_steps=steps,
            #                    path=path_save, output_file=f'S.Mises_{tra}')

            # mask_object = np.argwhere(n != NodeTypeDP.ACTUATOR).reshape(-1)

            # append error to global errors
            global_se_pos.append(np.array((predictions[:, :, :3] - targets[:, :, :3]) ** 2).reshape(-1))
            global_se_velocity.append(np.array((predictions[:, :, 3:6] - targets[:, :, 3:6]) ** 2).reshape(-1))
            global_se_energy.append(np.array((predictions[:, :, 6:] - targets[:, :, 6:]) ** 2).reshape(-1))
            # value error trajectory
            value_pos = np.mean(global_se_pos[-1])
            value_velocity = np.mean(global_se_velocity[-1])
            value_energy = np.mean(global_se_energy[-1])
            print(f'mse pos = {value_pos}')
            print(f'mse velocity = {value_velocity}')
            print(f'mse energy = {value_energy}')
            mse_pos.append(value_pos)
            mse_velocity.append(value_velocity)
            mse_energy.append(value_energy)
            # Store values in dictionaries for each trajectory
            mse_pos_dict[f'trajectory_{tra_name}'] = float(value_pos)
            mse_velocity_dict[f'trajectory_{tra_name}'] = float(value_velocity)
            mse_energy_dict[f'trajectory_{tra_name}'] = float(value_energy)

            # The corresponding Reltaive inf MSE
            global_se_inf_pos.append(se_inf(targets[:, :, :3], predictions[:, :, :3]).reshape(-1))
            global_se_inf_velocity.append(se_inf(targets[:, :, 3:6], predictions[:, :, 3:6]).reshape(-1))
            global_se_inf_energy.append(se_inf(targets[:, :, 6:], predictions[:, :, 6:]).reshape(-1))
            # value error trajectory
            value_pos = np.mean(global_se_inf_pos[-1])
            value_velocity = np.mean(global_se_inf_velocity[-1])
            value_energy = np.mean(global_se_inf_energy[-1])
            print(f'mse inf pos = {value_pos}')
            print(f'mse inf velocity = {value_velocity}')
            print(f'mse inf energy = {value_energy}')
            mse_inf_pos.append(value_pos)
            mse_inf_velocity.append(value_velocity)
            mse_inf_energy.append(value_energy)
            # Store values in dictionaries for each trajectory
            mse_inf_pos_dict[f'trajectory_{tra_name}'] = float(value_pos)
            mse_inf_velocity_dict[f'trajectory_{tra_name}'] = float(value_velocity)
            mse_inf_energy_dict[f'trajectory_{tra_name}'] = float(value_energy)

        rmse_pos = np.sqrt(np.mean(np.concatenate(global_se_pos)))
        rmse_velocity = np.sqrt(np.mean(np.concatenate(global_se_velocity)))
        rmse_energy = np.sqrt(np.mean(np.concatenate(global_se_energy)))

        rrmse_pos = np.sqrt(np.mean(np.concatenate(global_se_inf_pos)))
        rrmse_velocity = np.sqrt(np.mean(np.concatenate(global_se_inf_velocity)))
        rrmse_energy = np.sqrt(np.mean(np.concatenate(global_se_inf_energy)))

        print(f'SPLIT: {split}')
        print(f'ROLLOUT STEPS={steps} for NUM TRAJ={len(mse_pos_dict.keys())}')
        print(f'    RMSE positions: {rmse_pos}')
        print(f'    RMSE velocity: {rmse_velocity}')
        print(f'    RMSE energy: {rmse_energy}')
        print(f'    RRMSE positions: {rrmse_pos}')
        print(f'    RRMSE velocity: {rrmse_velocity}')
        print(f'    RRMSE energy: {rrmse_energy}')

        # Open the file in append mode ('a') to add new lines without overwriting the previous content
        with open(path_save+'/errors.txt', 'a') as file:
            # Write the error information to the file
            file.write(f'ROLLOUT STEPS={steps} for NUM TRAJ={len(mse_pos_dict.keys())}\n')
            file.write(f'    RMSE positions: {rmse_pos}\n')
            file.write(f'    RMSE velocity: {rmse_velocity}\n')
            file.write(f'    RMSE energy: {rmse_energy}\n')
            file.write(f'    RRMSE positions: {rrmse_pos}\n')
            file.write(f'    RRMSE velocity: {rrmse_velocity}\n')
            file.write(f'    RRMSE energy: {rrmse_energy}\n')

        # Combine the two dictionaries into a single dictionary
        data_to_save = {
            'metadata': {'steps': steps, 'tra': len(mse_pos_dict.keys())},
            'mse_positions': mse_pos_dict,
            'mse_velocityes': mse_velocity_dict,
            'mse_energies': mse_energy_dict,
            'mse_inf_positions': mse_inf_pos_dict,
            'mse_inf_velocityes': mse_inf_velocity_dict,
            'mse_inf_energies': mse_inf_energy_dict
        }
        # Save the dictionary as a JSON file
        json_path = path_save + f'/errors.json'
        with open(json_path, 'a') as json_file:
            json.dump(data_to_save, json_file, indent=4)  # Use indent=4 for pretty-printing

        # Extracting mse_positions and mse_velocity values from your data
        mse_pos = [value for key, value in data_to_save['mse_positions'].items()]
        mse_velocity = [value for key, value in data_to_save['mse_velocityes'].items()]
        mse_energy = [value for key, value in data_to_save['mse_energies'].items()]
        mse_inf_pos = [value for key, value in data_to_save['mse_inf_positions'].items()]
        mse_inf_velocity = [value for key, value in data_to_save['mse_inf_velocityes'].items()]
        mse_inf_energy = [value for key, value in data_to_save['mse_inf_energies'].items()]

        # Compute rmse for mse_positions
        rmse_pos = np.sqrt(np.array(mse_pos))
        rmse_velocity = np.sqrt(np.array(mse_velocity))
        rmse_energy = np.sqrt(np.array(mse_energy))
        rrmse_pos = np.sqrt(np.array(mse_inf_pos))
        rrmse_velocity = np.sqrt(np.array(mse_inf_velocity))
        rrmse_energy = np.sqrt(np.array(mse_inf_energy))

        # Compute the standard deviation and standard error for rmse_positions
        std_dev_rmse_pos = np.std(rmse_pos, ddof=1)
        n_rmse_pos = len(rmse_pos)
        standard_error_rmse_pos = std_dev_rmse_pos / np.sqrt(n_rmse_pos)
        # Compute the standard deviation and standard error for rmse_velocity
        std_dev_rmse_velocity = np.std(rmse_velocity, ddof=1)
        n_rmse_velocity = len(rmse_velocity)
        standard_error_rmse_velocity = std_dev_rmse_velocity / np.sqrt(n_rmse_velocity)
        # Compute the standard deviation and standard error for rmse_energy
        std_dev_rmse_energy = np.std(rmse_energy, ddof=1)
        n_rmse_energy = len(rmse_energy)
        standard_error_rmse_energy = std_dev_rmse_energy / np.sqrt(n_rmse_energy)

        # Compute the standard deviation and standard error for rrmse_positions
        std_dev_rrmse_pos = np.std(rrmse_pos, ddof=1)
        n_rrmse_pos = len(rrmse_pos)
        standard_error_rrmse_pos = std_dev_rrmse_pos / np.sqrt(n_rrmse_pos)
        # Compute the standard deviation and standard error for rrmse_velocity
        std_dev_rrmse_velocity = np.std(rrmse_velocity, ddof=1)
        n_rrmse_velocity = len(rmse_velocity)
        standard_error_rrmse_velocity = std_dev_rrmse_velocity / np.sqrt(n_rrmse_velocity)
        # Compute the standard deviation and standard error for rrmse_energy
        std_dev_rrmse_energy = np.std(rrmse_energy, ddof=1)
        n_rrmse_energy = len(rmse_energy)
        standard_error_rrmse_energy = std_dev_rrmse_energy / np.sqrt(n_rrmse_energy)

        # Output the standard errors
        print("SE of RMSE positions:", standard_error_rmse_pos, "std:", std_dev_rmse_pos)
        print("SE of RMSE velocity:", standard_error_rmse_velocity, "std:",std_dev_rmse_velocity)
        print("SE of RMSE energy:", standard_error_rmse_energy, "std:",std_dev_rmse_energy)
        print("SE of RRMSE positions:", standard_error_rrmse_pos, "std:", std_dev_rrmse_pos)
        print("SE of RRMSE velocity:", standard_error_rrmse_velocity, "std:", std_dev_rrmse_velocity)
        print("SE of RRMSE energy:", standard_error_rrmse_energy, "std:",std_dev_rrmse_energy)

        # Append the results to the text file
        with open(path_save+'/errors.txt', 'a') as file:  # Use 'a' for append mode
            file.write(f"    SE of RMSE positions: {standard_error_rmse_pos}\n")
            file.write(f"    SE of RMSE velocity: {standard_error_rmse_velocity}\n")
            file.write(f"    SE of RMSE energy: {standard_error_rmse_energy}\n")
            file.write(f"    SE of RRMSE positions: {standard_error_rrmse_pos}\n")
            file.write(f"    SE of RRMSE velocity: {standard_error_rrmse_velocity}\n")
            file.write(f"    SE of RRMSE energy: {standard_error_rrmse_energy}\n")


