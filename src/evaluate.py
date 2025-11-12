import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from amb.metrics import rrmse_inf
from torch_geometric.loader import DataLoader
from src.utils.utils import print_error, generate_folder
from src.utils.plots import plot_2D_image, plot_2D, plot_image3D, plotError, plot_3D, video_plot_3D, plot_3D_mp, plot_PyVista, plot_PyVista_comparativo
from src.utils.utils import compute_connectivity
from src.dataLoader.dataset import GraphDataset


def compute_error(z_net, z_gt, state_variables):
    # Compute error
    e = z_net.numpy() - z_gt.numpy()
    gt = z_gt.numpy()

    error = {clave: [] for clave in state_variables}
    L2_list = {clave: [] for clave in state_variables}

    for i, sv in enumerate(state_variables):
        L2 = ((e[1:, :, i] ** 2).sum(1) / (gt[1:, :, i] ** 2).sum(1)) ** 0.5
        L22 = np.mean(((e[1:, :, i] ** 2).sum(1) / (gt[1:, :, i] ** 2).sum(1))) ** 0.5
        error[sv] = L22
        L2_list[sv].extend(L2)

    return error, L2_list


def roll_out(nodal_gnn, dataloader, device, radius_connectivity, dtset_type, glass_flag=False):
    data = [sample for sample in dataloader]
    cnt_conet = 0
    cnt_gnn = 0

    dim_z = data[0].x.shape[1]
    N_nodes = data[0].x.shape[0]
    if glass_flag:
        n = torch.zeros(len(data) + 1, N_nodes)
        n[0] = data[0].n
        N_nodes = data[0].x[n[0] == 1, :].shape[0]
    z_net = torch.zeros(len(data) + 1, N_nodes, dim_z)
    z_gt = torch.zeros(len(data) + 1, N_nodes, dim_z)

    # Initial conditions
    if glass_flag:
        z_net[0] = data[0].x[n[0] == 1]
        z_gt[0] = data[0].x[n[0] == 1]
    else:
        z_net[0] = data[0].x
        z_gt[0] = data[0].x

    z_denorm = data[0].x
    edge_index = data[0].edge_index

    try:
        for t, snap in enumerate(data):
            snap.x = z_denorm
            snap.edge_index = edge_index
            snap = snap.to(device)
            with torch.no_grad():
                start_time = time.time()
                z_denorm, z_t1, _ = nodal_gnn.predict_step(snap, 1)
                cnt_gnn += time.time() - start_time
            if dtset_type == 'fluid':
                pos = z_denorm[:, :3].clone()
                start_time = time.time()
                edge_index = compute_connectivity(np.asarray(pos.cpu()), radius_connectivity, add_self_edges=False).to(
                    device)
                cnt_conet += time.time() - start_time
            else:
                edge_index = snap.edge_index
            if glass_flag:
                z_net[t + 1] = z_denorm[snap.n == 1]
                z_gt[t + 1] = z_t1[snap.n == 1]
            else:
                z_net[t + 1] = z_denorm
                z_gt[t + 1] = z_t1
    except:
        print(f'Ha fallado el rollout en el momento: {t}')

    print(f'El tiempo tardado en el compute connectivity: {cnt_conet}')
    print(f'El tiempo tardado en la red: {cnt_gnn}')
    return z_net, z_gt, t+1, snap.plot_info[0][0], snap.plot_info[0][1]


def generate_results(plasticity_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):
    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_gif_pdc = os.path.join(output_dir_exp, f'result_pdc.gif')
    save_dir_gif_pyvista = os.path.join(output_dir_exp, f'result_pyvista.gif')

    # Make roll out
    start_time = time.time()
    z_net, z_gt, t, celulas, conectividad = roll_out(plasticity_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'],
                              dInfo['dataset']['type'])
    print(f'El tiempo tardado en el rollout: {time.time() - start_time}')
    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        error, L2_list = compute_error(z_net[1:, :, :], z_gt[1:, :, :], dInfo['dataset']['state_variables'])
        lines = print_error(error)
        f.write('\n'.join(lines))
        print("[Test Evaluation Finished]\n")
        f.close()
    plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'], output_dir_exp)

    if dInfo['project_name'] == 'Beam_2D':
        plot_2D_image(z_net, z_gt, -1, 4, output_dir=output_dir_exp)
        plot_2D(z_net, z_gt, save_dir_gif, var=4)
    else:
        # video_plot_3D(z_net, z_gt, save_dir=save_dir_gif_pdc)
        # plot_3D(z_net, z_gt, save_dir=save_dir_gif, var=-1)
        plot_PyVista_comparativo(z_net, z_gt, celulas, conectividad, save_dir_gif_pyvista, var=6)



def  generate_results_recons(gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):
    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_gif_pdc = os.path.join(output_dir_exp, f'result_pdc.gif')
    save_dir_gif_pyvista = os.path.join(output_dir_exp, f'result_pyvista.gif')

    # Make roll out
    start_time = time.time()

    data = [sample for sample in test_dataloader]
    snap = data[0].to(device)
    z_net, z_t1, _ = gnn.predict_step(snap, 1)
    z_gt = data[0].y[snap.n == 1, :].cpu().numpy()
    z_net =z_net[snap.n == 1, :].cpu().numpy()
    test_sample = data[0]

    pos_x, pos_y, pos_z, vel_x_gt, vel_y_gt, vel_z_gt, e_gt = data[0].y[snap.n == 1, :].cpu().numpy().T
    # pos_x, pos_y, pos_z, vel_x_gt, vel_y_gt, vel_z_gt, e_gt = z_net[snap.n == 1, :].cpu().numpy().T
    _, _, _, vel_x_net, vel_y_net, vel_z_net, e_net = z_net.T


    # Crear figura con una sola fila y tres columnas
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot Velocity X
    axes[0, 0].scatter(pos_x, vel_x_net, s=1, color="blue", label="Predicted")
    axes[0, 0].scatter(pos_x, vel_x_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[0, 0].set_xlabel("Position X")
    axes[0, 0].set_ylabel("Velocity X")
    axes[0, 0].set_title("VELOCITY vs Position X")
    axes[0, 0].legend()

    # Plot Velocity Y
    axes[0, 1].scatter(pos_y, vel_y_net, s=1, color="blue", label="Predicted")
    axes[0, 1].scatter(pos_y, vel_y_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[0, 1].set_xlabel("Position Y")
    axes[0, 1].set_ylabel("Velocity Y")
    axes[0, 1].set_title("Velocity Y vs Position Y")
    axes[0, 1].legend()

    # Plot Velocity Z
    axes[0, 2].scatter(pos_z, vel_z_net, s=1, color="blue", label="Predicted")
    axes[0, 2].scatter(pos_z, vel_z_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[0, 2].set_xlabel("Position Z")
    axes[0, 2].set_ylabel("Velocity Z")
    axes[0, 2].set_title("Velocity Z vs Position Z")
    axes[0, 2].legend()

    # Plot Velocity Y vs Position X
    axes[1, 0].scatter(pos_x, vel_y_net, s=1, color="blue", label="Predicted")
    axes[1, 0].scatter(pos_x, vel_y_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[1, 0].set_xlabel("Position X")
    axes[1, 0].set_ylabel("Velocity Y")
    axes[1, 0].set_title("Velocity Y vs Position X")
    axes[1, 0].legend()

    # Plot Velocity Y vs Position Y
    axes[1, 1].scatter(pos_y, vel_y_net, s=1, color="blue", label="Predicted")
    axes[1, 1].scatter(pos_y, vel_y_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[1, 1].set_xlabel("Position Y")
    axes[1, 1].set_ylabel("Velocity Y")
    axes[1, 1].set_title("Velocity Y vs Position Y")
    axes[1, 1].legend()

    # Plot Velocity Y vs Position Z
    axes[1, 2].scatter(pos_z, vel_y_net, s=1, color="blue", label="Predicted")
    axes[1, 2].scatter(pos_z, vel_y_gt, s=1, color="red", label="Ground Truth", alpha=0.5)
    axes[1, 2].set_xlabel("Position Z")
    axes[1, 2].set_ylabel("Velocity Y")
    axes[1, 2].set_title("Velocity Y vs Position Z")
    axes[1, 2].legend()

    plt.tight_layout()

    # Guardar imagen
    file_path = os.path.join(output_dir_exp, "velocities.png")
    plt.savefig(file_path)
    plt.close(fig)

    state_variables = dInfo['dataset']['state_variables']
    e = z_net - z_gt
    gt = z_gt

    error = {clave: [] for clave in state_variables}
    L2_list = {clave: [] for clave in state_variables}

    for i, sv in enumerate(state_variables):
        L2 = ((e[ :, i] ** 2).sum() / (gt[:, i] ** 2).sum()) ** 0.5
        L22 = np.mean(((e[ :, i] ** 2).sum() / (gt[ :, i] ** 2).sum())) ** 0.5
        print(f'{sv}  L2: {L2}')

        error[sv] = L22
        L2_list[sv].extend([L2])

    print(f'El tiempo tardado en el rollout: {time.time() - start_time}')
    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        error, L2_list = compute_error(z_net, z_gt, dInfo['dataset']['state_variables'])
        lines = print_error(error)
        f.write('\n'.join(lines))
        print("[Test Evaluation Finished]\n")
        f.close()
    plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'], output_dir_exp)

    if dInfo['project_name'] == 'Beam_2D':
        plot_2D_image(z_net, z_gt, -1, 4, output_dir=output_dir_exp)
        plot_2D(z_net, z_gt, save_dir_gif, var=4)
    else:
        # video_plot_3D(z_net, z_gt, save_dir=save_dir_gif_pdc)
        # plot_3D(z_net, z_gt, save_dir=save_dir_gif, var=-1)
        plot_PyVista_comparativo(z_net, z_gt, celulas, conectividad, save_dir_gif_pyvista, var=6)

