import os
import time
import torch
import numpy as np
from src.utils.utils import print_error, generate_folder
from src.utils.plots import plot_2D_image, plot_2D, video_plot_3D, plotError, plot_3D
from src.utils.utils import compute_connectivity


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


def roll_out(plasticity_gnn, dataloader, device, radius_connectivity, dim_data, dtset_type, glass_flag=False):
    data = [sample for sample in dataloader]
    cnt_conet = 0
    cnt_gnn = 0

    dim_z = data[0].x.shape[1]
    N_nodes = data[0].x.shape[0]
    if glass_flag:
        n = torch.zeros(len(data) + 1, N_nodes)
        n[0] = data[0].n
        N_nodes = data[0].x[n[0]==1,:].shape[0]
    z_net = torch.zeros(len(data) + 1, N_nodes, dim_z)
    z_gt = torch.zeros(len(data) + 1, N_nodes, dim_z)

    # Initial conditions
    if glass_flag:
        z_net[0] = data[0].x[n[0]==1]
        z_gt[0] = data[0].x[n[0]==1]
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
                z_denorm, z_t1 = plasticity_gnn.predict_step(snap, 1)
                cnt_gnn += time.time()-start_time
            pos = z_denorm[:, :3].clone()
            if dim_data == 2:
                pos[:, 2] = pos[:, 2] * 0
            if dtset_type=='fluid':
                start_time = time.time()
                edge_index = compute_connectivity(np.asarray(pos.cpu()), radius_connectivity, add_self_edges=False).to(device)
                cnt_conet += time.time()-start_time
            else:
                edge_index = snap.edge_index
            if glass_flag:
                z_net[t + 1] = z_denorm[snap.n==1]
                z_gt[t + 1] = z_t1[snap.n==1]
            else:
                z_net[t + 1] = z_denorm
                z_gt[t + 1] = z_t1
    except:
        print(f'Ha fallado el rollout en el momento: {t}')

    print(f'El tiempo tardado en el compute connectivity: {cnt_conet}')
    print(f'El tiempo tardado en la red: {cnt_gnn}')
    return z_net, z_gt, t
def generate_results(plasticity_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):

    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_pcd = os.path.join(output_dir_exp, f'result_pdc.gif')
    dim_data = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
    # Make roll out
    start_time = time.time()
    z_net, z_gt, t = roll_out(plasticity_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'], dim_data, dInfo['dataset']['type'])
    print(f'El tiempo tardado en el rollout: {time.time()-start_time}')
    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        error, L2_list = compute_error(z_net[1:, :, :], z_gt[1:, :, :], dInfo['dataset']['state_variables'])
        lines = print_error(error)
        f.write('\n'.join(lines))
        print("[Test Evaluation Finished]\n")
        f.close()
    plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'], output_dir_exp)

    if dInfo['dataset']['dataset_dim'] == '2D':
        plot_2D_image(z_net, z_gt, -1, 4, output_dir=output_dir_exp)
        plot_2D(z_net, z_gt, save_dir_gif, var=4)
    else:
        data = [sample for sample in test_dataloader]
        video_plot_3D(z_net[:t, :, :], z_gt[:t, :, :], save_dir=save_dir_pcd,)
        # plot_image3D(z_net[:t, :, :], z_gt[:t, :, :], output_dir_exp, var=-1, step=-1, n=data[0].n[:,0])
        # plot_image3D(z_net, z_gt, output_dir_exp, var=2, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=1, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=4, step=70, n=data[0].n)
        plot_3D(z_net[:t, :, :], z_gt[:t, :, :], save_dir=save_dir_gif, var=-1)

