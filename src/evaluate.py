import os
import time
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# from amb.metrics import rrmse_inf
from torch_geometric.loader import DataLoader
from src.utils.utils import print_error, generate_folder, compute_connectivity
from src.utils.plots import plot_2D_image, plot_2D, plot_image3D, plotError, plot_3D, video_plot_3D, plot_3D_mp, plot_PyVista, plot_PyVista_comparativo, plot_velPos_gnn, plot_velocity_3D, plot_flow_comparison
from src.dataLoader.dataset import GraphDataset
from torch_geometric.nn import radius_graph


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
                edge_index = radius_graph(pos, r=radius_connectivity, loop=False, flow='source_to_target', max_num_neighbors=1000)
                # edge_index = compute_connectivity(np.asarray(pos.cpu()), radius_connectivity, add_self_edges=False).to(
                #     device)
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
    return z_net, z_gt, t+1 #, snap.plot_info[0][0], snap.plot_info[0][1]


def generate_results(plasticity_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):
    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_gif_pdc = os.path.join(output_dir_exp, f'result_pdc.gif')
    save_dir_gif_pyvista = os.path.join(output_dir_exp, f'result_pyvista.gif')

    # Make roll out
    start_time = time.time()
    z_net, z_gt, t = roll_out(plasticity_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'],
                              dInfo['dataset']['type'])  #celulas, conectividad 
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
        plot_3D(z_net, z_gt, save_dir=save_dir_gif, var=-1)
        # plot_PyVista_comparativo(z_net, z_gt, celulas, conectividad, save_dir_gif_pyvista, var=6)



def  generate_results_recons_1sample(gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):
    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_gif_pdc = os.path.join(output_dir_exp, f'result_pdc.gif')
    save_dir_gif_pyvista = os.path.join(output_dir_exp, f'result_pyvista.gif')

    start_time = time.time()

    data = [sample for sample in test_dataloader]
    snap = data[0].to(device)
    z_net, z_t1, _ = gnn.predict_step(snap, 1)
    z_gt = data[0].y[snap.n == 1, :].cpu().numpy()
    z_net =z_net[snap.n == 1, :].cpu().numpy()
    test_sample = data[0]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection="3d")
    sc = ax.scatter(z_gt[:, 0], z_gt[:, 2], z_gt[:, 1],
                s=8, alpha=0.8, c=z_net[:, 4])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")    
    ax.set_box_aspect([1, 1, 1])
    plt.colorbar(sc, ax=ax)


    z_gt = data[0].y[snap.n == 1, :].cpu().numpy()
    i = 0
    file_path = os.path.join(output_dir_exp, f'{str(i)}_velocities.png')

    plot_velPos_gnn(z_gt, z_net, file_path)


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


def  generate_results_recons(gnn, trainer, test_dataloader, dInfo, scaler, output_dir_exp, pahtDInfo, pathWeights):
    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_gif_pdc = os.path.join(output_dir_exp, f'result_pdc.gif')
    save_dir_gif_pyvista = os.path.join(output_dir_exp, f'result_pyvista.gif')

    start_time = time.time()

    preds = trainer.predict(gnn, dataloaders=test_dataloader)

    for i, pred in enumerate(preds):    
        z_net, z_gt, _, n = pred

        z_net =z_net[n == 1, :].cpu().numpy()
        z_gt =z_gt[n == 1, :].cpu().numpy() 

        plot_flow_comparison(z_gt, z_net, n_variable=4, file_path=os.path.join(output_dir_exp, f'{str(i)}_velocities3D.png'))
        plot_velPos_gnn(z_gt, z_net, os.path.join(output_dir_exp, f'{str(i)}_velocities.png'))

    plot_velocity_3D(z_gt, z_net)

    all_z_net = torch.cat([p[0][:,3:] for p in preds], dim=0)
    all_z_gt = torch.cat([p[1][:,3:] for p in preds], dim=0)


    metrics = compute_fluid_metrics(all_z_net, all_z_gt, scaler[0])

     
    with open(os.path.join(output_dir_exp, 'metrics.json'), "w") as f:
        json.dump(metrics, f, indent=4)



def compute_fluid_metrics_(y_hat, y, scaler):
    eps = 1e-12
    metrics = {}

    # ----------------------------------------------
    # 1) MAE y RMSE (seguro, estable, físico)
    # ----------------------------------------------
    e = y_hat - y
    metrics["MAE_vx"], metrics["MAE_vy"], metrics["MAE_vz"], metrics["MAE_E"] = torch.mean(torch.abs(e), dim=0).tolist()
    metrics["RMSE_vx"], metrics["RMSE_vy"], metrics["RMSE_vz"], metrics["RMSE_E"] = torch.sqrt(torch.mean(e**2, dim=0)).tolist()

    # ----------------------------------------------
    # 2) Error relativo basado en rango físico
    # (sin divisiones explosivas)
    # ----------------------------------------------
    v_range = (scaler.data_max_[3:6].max() - scaler.data_min_[3:6].min())
    metrics["Rel_vx_range"] = torch.mean(torch.abs(e[:,0]) / v_range).item()
    metrics["Rel_vy_range"] = torch.mean(torch.abs(e[:,1]) / v_range).item()
    metrics["Rel_vz_range"] = torch.mean(torch.abs(e[:,2]) / v_range).item()

    metrics["Rel_E_range"] = torch.mean(torch.abs(e[:,3]) / scaler.data_max_[6]).item()

    # ----------------------------------------------
    # 3) Error vectorial: magnitud y dirección
    # ----------------------------------------------
    v_hat = y_hat[:, :3]
    v_gt = y[:, :3]

    mag_hat = torch.norm(v_hat, dim=1)
    mag_gt = torch.norm(v_gt, dim=1)
    mask = mag_gt > mag_gt.max()*0.25

    metrics["MAE_speed"] = torch.mean(torch.abs(mag_hat - mag_gt)).item()
    metrics["RMSE_speed"] = torch.sqrt(torch.mean((mag_hat - mag_gt)**2)).item()

    # ----------------------------------------------
    # 4) Error angular (seguro y estable)
    # ----------------------------------------------
    dot = (v_hat[mask] * v_gt[mask]).sum(dim=1)
    denom = (mag_hat[mask] * mag_gt[mask] + eps)
    cosine = torch.clamp(dot / denom, -1 + eps, 1 - eps)

    angle = torch.acos(cosine)
    metrics["mean_angle_deg"] = (angle.mean() * 180 / torch.pi).item()
    metrics["median_angle_deg"] = (angle.median() * 180 / torch.pi).item()
    metrics["angle_points_used"] = int(mask.sum())

    # ----------------------------------------------
    # 5) Conservación global (energía total)
    # ----------------------------------------------
    E_pred = y_hat[:,3].sum()
    E_true = y[:,3].sum()

    metrics["Energy_rel_error_total"] = torch.abs(E_pred - E_true).item() / (E_true.item() + eps)

    return metrics


def compute_fluid_metrics(y_hat, y, scaler):
    eps = 1e-12
    metrics = {}
    
    # Error bruto para boxplots y desviaciones
    e = y_hat - y
    abs_e = torch.abs(e)
    
    # 1) MAE y RMSE con Desviación Estándar
    for i, var in enumerate(['vx', 'vy', 'vz', 'E']):
        metrics[f"MAE_{var}"] = torch.mean(abs_e[:, i]).item()
        metrics[f"STD_{var}"] = torch.std(abs_e[:, i]).item() # Importante para la tesis
        metrics[f"RMSE_{var}"] = torch.sqrt(torch.mean(e[:, i]**2)).item()

    # 2) nRMSE (Normalizado por el rango) - Muy común en papers de GNN
    v_range = (scaler.data_max_[3:6] - scaler.data_min_[3:6])
    v_range = torch.tensor(v_range, device=y.device)
    for i, var in enumerate(['vx', 'vy', 'vz']):
        metrics[f"nRMSE_{var}"] = metrics[f"RMSE_{var}"] / (v_range[i] + eps)

    # 3) Error Angular (Solo donde hay movimiento significativo)
    mag_gt = torch.norm(y[:, :3], dim=1)
    mask = mag_gt > (torch.max(mag_gt) * 0.1) # Umbral del 10%
    
    dot = torch.sum(y_hat[mask, :3] * y[mask, :3], dim=1)
    denom = torch.norm(y_hat[mask, :3], dim=1) * mag_gt[mask] + eps
    angles = torch.acos(torch.clamp(dot / denom, -1.0 + eps, 1.0 - eps))
    
    metrics["Mean_Angle_Deg"] = torch.rad2deg(torch.mean(angles)).item()
    metrics["Std_Angle_Deg"] = torch.rad2deg(torch.std(angles)).item()

    # 4) Conservación de Energía Global
    E_total_pred = torch.sum(y_hat[:, 3])
    E_total_gt = torch.sum(y[:, 3])
    metrics["Global_Energy_Rel_Err"] = torch.abs(E_total_pred - E_total_gt).item() / (torch.abs(E_total_gt) + eps)

    return metrics