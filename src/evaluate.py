import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from amb.metrics import rrmse_inf
from torch_geometric.loader import DataLoader
from src.utils.utils import print_error, generate_folder
from src.utils.plots import plot_2D_image, plot_2D, plot_image3D, plotError, plot_3D, video_plot_3D
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
                z_denorm, z_t1 = nodal_gnn.predict_step(snap, 1)
                cnt_gnn += time.time()-start_time
            if dtset_type=='fluid':
                pos = z_denorm[:, :3].clone()
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


def roll_out_false(nodal_gnn, dataloader, device):
    data = [sample for sample in dataloader]
    cnt_conet = 0
    cnt_gnn = 0

    dim_z = data[0].x.shape[1]
    N_nodes = data[0].x.shape[0]

    z_net = torch.zeros(len(data), N_nodes, dim_z)
    z_gt = torch.zeros(len(data), N_nodes, dim_z)

    try:
        for t, snap in enumerate(data):
            snap = snap.to(device)
            with torch.no_grad():
                start_time = time.time()
                y_pred, z_t1 = nodal_gnn.predict_step(snap, 1)
                cnt_gnn += time.time()-start_time

            z_net[t + 1] = y_pred
            z_gt[t + 1] = snap.y
    except:
        print(f'Ha fallado el rollout en el momento: {t}')

    print(f'El tiempo tardado en el compute connectivity: {cnt_conet}')
    print(f'El tiempo tardado en la red: {cnt_gnn}')
    return z_net, z_gt
def generate_results(plasticity_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):

    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    save_dir_gif_pdc = os.path.join(output_dir_exp, f'result_pdc.gif')
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
    # plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'], output_dir_exp)

    if dInfo['project_name'] == 'Beam_2D':
        plot_2D_image(z_net, z_gt, -1, 4, output_dir=output_dir_exp)
        plot_2D(z_net, z_gt, save_dir_gif, var=4)
    else:
        # data = [sample for sample in test_dataloader]
        video_plot_3D(z_net, z_gt, save_dir=save_dir_gif_pdc,)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=-1, step=-1, n=data[0].n[:,0])
        # plot_image3D(z_net, z_gt, output_dir_exp, var=2, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=1, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=4, step=70, n=data[0].n)
        plot_3D(z_net, z_gt, save_dir=save_dir_gif, var=-1)


def set_equal_aspect_3d(ax):
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    ranges = np.ptp(limits, axis=1)
    max_range = ranges.max()
    centers = np.mean(limits, axis=1)
    new_limits = np.array([centers - max_range / 2, centers + max_range / 2]).T
    ax.set_xlim(new_limits[0])
    ax.set_ylim(new_limits[1])
    ax.set_zlim(new_limits[2])
def generate_results2(plasticity_gnn, test_dataloader, dInfo, device, output_dir_exp, pahtDInfo, pathWeights):

    # Generate output folder
    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    save_dir_gif = os.path.join(output_dir_exp, f'result.gif')
    dim_data = 2 if dInfo['dataset']['dataset_dim'] == '2D' else 3
    # Make roll out
    start_time = time.time()
    z_net, z_gt= roll_out_false(plasticity_gnn, test_dataloader, device)
    print(f'El tiempo tardado en el rollout: {time.time()-start_time}')

    data = [sample for sample in test_dataloader]
    # for i in range(len(data)):
    for c in range(6):
        i = 19
        x = data[i]
        s_color = c
        # # Crear una figura 3D
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(121, projection='3d')
        scatter = ax.scatter(x.x[:,0], x.x[:,2], x.x[:,1], c=z_net[i, :, s_color], cmap='viridis', marker='o')
        fig.colorbar(scatter, ax=ax, label='Valores')
        set_equal_aspect_3d(ax)

        ax = fig.add_subplot(122, projection='3d')
        scatter = ax.scatter(x.x[:, 0], x.x[:, 2], x.x[:, 1], c=z_gt[i, :, s_color], cmap='viridis', marker='o')
        fig.colorbar(scatter, ax=ax, label='Valores')
        set_equal_aspect_3d(ax)
        plt.savefig(os.path.join(output_dir_exp, f'{dInfo["dataset"]["state_variables_out"][c]}.png'))
        # plt.show()


    filePath = os.path.join(output_dir_exp, 'metrics.txt')
    with open(filePath, 'w') as f:
        error, L2_list = compute_error(z_net[1:, :, :], z_gt[1:, :, :], dInfo['dataset']['state_variables_out'])
        lines = print_error(error)
        f.write('\n'.join(lines))
        print("[Test Evaluation Finished]\n")
        f.close()
    # plotError(z_gt, z_net, L2_list, dInfo['dataset']['state_variables'], dInfo['dataset']['dataset_dim'], output_dir_exp)

    if dInfo['project_name'] == 'Beam_2D':
        plot_2D_image(z_net, z_gt, -1, 4, output_dir=output_dir_exp)
        plot_2D(z_net, z_gt, save_dir_gif, var=4)
    else:
        data = [sample for sample in test_dataloader]
        video_plot_3D(z_net, z_gt, save_dir=save_dir_gif_pdc,)
        # plot_image3D(z_net, z_g[t, output_dir_exp, var=-1, step=-1, n=data[0].n[:,0])
        # plot_image3D(z_net, z_gt, output_dir_exp, var=2, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=1, step=70, n=data[0].n)
        # plot_image3D(z_net, z_gt, output_dir_exp, var=4, step=70, n=data[0].n)
        plot_3D(z_net, z_gt, save_dir=save_dir_gif, var=-1)




def compute_(nodal_gnn, dset_dir, dInfo, device, output_dir_exp, pahtDInfo, pathWeights, project_name):

    output_dir_exp = generate_folder(output_dir_exp, pahtDInfo, pathWeights)
    Linf_q = []
    Linf_v = []
    Linf_e = []
    error = dict({'q': [], 'v': [], 'sigma': [], 'q_inf': [], 'v_inf': [], 'e_inf': []})
    dset_dir = dInfo['dataset']['test_folder']
    for datasetTestPath in os.listdir(dset_dir):
        test_set = GraphDataset(dInfo, os.path.join(dset_dir, datasetTestPath), length=40)
        test_dataloader = DataLoader(test_set, batch_size=1)

        # Compute Simulations
        z_net, z_gt, q_0 = roll_out(nodal_gnn, test_dataloader, device, dInfo['dataset']['radius_connectivity'],
                               dInfo['dataset']['type'])

        # Compute error
        e = z_net[1:].numpy() - z_gt[1:].numpy()
        gt = z_gt[1:].numpy()

        # Position + Velocity + Stress Tensor
        if project_name == 'Beam_3D':
            L2_q = ((e[:, :, 0:3] ** 2).sum((1, 2)) / (gt[:, :, 0:3] ** 2).sum((1, 2))) ** 0.5
            L2_v = ((e[:, :, 3:6] ** 2).sum((1, 2)) / (gt[:, :, 3:6] ** 2).sum((1, 2))) ** 0.5
            L2_sigma = ((e[:, :, 6:] ** 2).sum((1, 2)) / (gt[:, :, 6:] ** 2).sum((1, 2))) ** 0.5
        elif project_name == 'Glass3D':
            L2_q = ((e[:, :, 0:3] ** 2).sum((1, 2)) / (gt[:, :, 0:3] ** 2).sum((1, 2))) ** 0.5
            L2_v = ((e[:, :, 3:6] ** 2).sum((1, 2)) / (gt[:, :, 3:6] ** 2).sum((1, 2))) ** 0.5
            L2_sigma = ((e[:, :, -1] ** 2).sum(1) / (gt[:, :, -1] ** 2).sum(1)) ** 0.5
            Linf_q_, Ltot_q = rrmse_inf(z_gt[:, :, 0:3], z_net[:, :, 0:3])
            Linf_v_, Ltot_v = rrmse_inf(z_gt[:, :, 3:6], z_net[:, :, 3:6])
            Linf_e_, Ltot_e = rrmse_inf(z_gt[:, :, 6:7], z_net[:, :, 6:7])
        elif project_name == 'Difusion_1D':
            L2_q = ((e[:, :, 0:1] ** 2).sum((1, 2)) / (gt[:, :, 0:1]+0.00008 ** 2).sum((1, 2))) ** 0.5
            L2_v = ((e[:, :, 1:] ** 2).sum((1, 2)) / (gt[:, :, 1:]+0.00008 ** 2).sum((1, 2))) ** 0.5
            Linf_q_, Ltot_q = rrmse_inf(z_gt[:, :, 0:1], z_net[:, :, 0:1])
            Linf_v_, Ltot_v = rrmse_inf(z_gt[:, :, 1:], z_net[:, :, 1:])
            Linf_e_, L2_sigma, Ltot_e = 0, L2_q*0, L2_q*0
        else:
            L2_q = ((e[:, :, 0:2] ** 2).sum((1, 2)) / (gt[:, :, 0:3] ** 2).sum((1, 2))) ** 0.5
            L2_v = ((e[:, :, 2:4] ** 2).sum((1, 2)) / (gt[:, :, 3:6] ** 2).sum((1, 2))) ** 0.5
            L2_sigma = ((e[:, :, 4:] ** 2).sum((1, 2)) / (gt[:, :, 6:] ** 2).sum((1, 2))) ** 0.5
        error['q'].extend(list(L2_q))
        error['v'].extend(list(L2_v))
        error['sigma'].extend(list(L2_sigma))
        # error['q_inf'].extend(list(Ltot_q))
        # error['v_inf'].extend(list(Ltot_v))
        # error['e_inf'].extend(list(Ltot_e))

        Linf_q.append(Linf_q_)
        Linf_v.append(Linf_v_)
        Linf_e.append(Linf_e_)

    # data = [error['q'], error['v'], error['sigma'], error['q_inf'], error['v_inf'], error['e_inf']]
    data = [error['q'], error['v'], error['sigma']]

    df = pd.DataFrame(data)
    df.to_csv('datos_nodal.csv', index=False, header=False)

    df_gnn = pd.read_csv('datos_gnn.csv', header=None)
    df_globalq = pd.read_csv('datos_gnn.csv', header=None)
    df_global = pd.read_csv('datos_nodal.csv', header=None)
    df_nodal = pd.read_csv('datos_nodal.csv', header=None)
    df_gnn = df_gnn.to_numpy()
    df_globalq = df_globalq.to_numpy()
    df_global = df_global.to_numpy()
    df_nodal = df_nodal.to_numpy()

    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#8EC5FC', '#FFD166', '#6A4C93']

    # Trazar los rectángulos de cada grupo
    plt.boxplot(df_gnn[0, :], positions=[0.8], boxprops=dict(color=colors[0]))
    plt.boxplot(df_globalq[0, :], positions=[1], boxprops=dict(color=colors[1]))
    plt.boxplot(df_global[0, :], positions=[1.2], boxprops=dict(color=colors[2]))
    plt.boxplot(df_nodal[0, :], positions=[1.4], boxprops=dict(color=colors[3]))
    plt.boxplot(df_gnn[1, :], positions=[1.8], boxprops=dict(color=colors[0]))
    plt.boxplot(df_globalq[1, :], positions=[2], boxprops=dict(color=colors[1]))
    plt.boxplot(df_global[1, :], positions=[2.2], boxprops=dict(color=colors[2]))
    plt.boxplot(df_nodal[1, :], positions=[2.4], boxprops=dict(color=colors[3]))
    plt.boxplot(df_gnn[2, :], positions=[2.8], boxprops=dict(color=colors[0]))
    plt.boxplot(df_globalq[2, :], positions=[3], boxprops=dict(color=colors[1]))
    plt.boxplot(df_global[2, :], positions=[3.2], boxprops=dict(color=colors[2]))
    plt.boxplot(df_nodal[2, :], positions=[3.4], boxprops=dict(color=colors[3]))
    # Configurar el eje y en escala logarítmica
    plt.yscale('log')
    # Personalización
    plt.title('GNN VS Global (Querqus) VS Nodal approach')
    # plt.xlabel('Grupo')
    plt.ylabel('Relative L2 Error')
    if project_name == 'Beam_3D':
        plt.xticks([1, 2, 3], ['Position', 'Velocity', 'Stress Tensor'])
    else:
        plt.xticks([1, 2, 3], ['Position', 'Velocity', 'Energy'])
    # plt.xticks([1, 2, 3], ['Position', 'Velocity', 'Energy'])

    legend_handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]
    plt.legend(legend_handles, ['Gnn', 'Globalq', 'Global', 'Nodal'])
    # Mostrar el gráfico
    plt.grid(True)
    # plt.show()
    plt.savefig('GNN VS Global (Querqus) VS Nodal approach.svg')

    return error


# plt.figure()
# colors = ['#FF6B6B', '#8EC5FC', '#FFD166', '#6A4C93']
# # Trazar los rectángulos de cada grupo
# plt.boxplot(df_gnn[0, :], positions=[1], boxprops=dict(color=colors[0]))
# plt.boxplot(df_gnn[1, :], positions=[2], boxprops=dict(color=colors[2]))
# plt.yscale('log')
# plt.ylabel('Relative L2 Error')
# plt.title('Local TI-GNN approach')
# plt.xticks([1, 2], ['Temperature', 'Velocity'])
# plt.grid(True)
# plt.show()