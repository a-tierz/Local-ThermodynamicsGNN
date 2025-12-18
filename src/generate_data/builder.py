import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import datetime
import random
import cv2
import shutil
import imageio
import open3d as o3d
from matplotlib.colors import Normalize
from torch.utils.data import Dataset
from src.dataLoader.reader import DataReader
from src.utils import compute_connectivity
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from torch_geometric.nn import radius_graph


class GraphDataset(Dataset):
    def __init__(self, path_to_data, sim_type, simulation_names=None,
                 dataset_name='', train_flag=True, save_csv=False,
                 desired_regions='all'):
        """
        Initialize the CustomDataset.

        Args:
            path_to_data (str): The directory containing your dataset.
            simulation_names (str, optional): simulation names of interest to load.
        """

        # Load data from generated files from Abaqus
        databuilder = DataBuilder(path_to_data, dataset_name, desired_regions=desired_regions)
        # Load the nodal variables and edges data
        print(f'Building data from simulations --> {simulation_names}')
        databuilder.get_nodal_variables(sim_type, desired_simulations=simulation_names, save_csv=save_csv)
        if sim_type == 'Glass':
            self.data_total = databuilder.calculate_glass_dataset(train_flag=train_flag)
        elif sim_type == 'Glass_recons':
            self.data_total = databuilder.calculate_glassrecons_dataset(train_flag=train_flag)

        # save dataset in .pt format
        databuilder.save_dataset(self.data_total, simulation_names=simulation_names, train_flag=train_flag)


class DataBuilder(object):
    """
    Class with a load method for loading data from ".txt" files within subfolders of a specified data folder.
    It allows you to specify desired simulations and reads the data using the DataReader class. The results are
    stored in a pandas DataFrame, and a 'Simulation' column is added to indicate the simulation name.
    Finally, the individual DataFrames are concatenated into a single dataset DataFrame.
    """

    def __init__(self, path_to_data_folder, dataset_name, train_split=0.9, desired_regions='all'):
        # Constructor to initialize the DataLoader with the path to the data folder
        self.path_to_data_folder = Path(path_to_data_folder)
        self.dataset_df = None  # Initialize a variable to store the final dataset DataFrame
        self.dataset_name = dataset_name
        self.train_split = train_split
        self.desired_regions = desired_regions
        self.radius_connectivity = 0.012 # Updated for lower particle density
        self.start_step = 18*2 #9
        self.sampling_dt = 0

    def get_nodal_variables(self, sim_type, desired_simulations=None, save_csv=False):
        # Method to load nodal variable data from text files

        print('Saving nodal variables data...')

        # Loop through all ".txt" files within subfolders of the data folder
        for file in self.path_to_data_folder.rglob('**/*data.txt'):

            # Extract the simulation name from the file path
            simulation_name = file.parts[-2]
            print(simulation_name)
            csv_path = os.path.join(os.path.dirname(file), f'{simulation_name}.csv')

            if save_csv or not os.path.isfile(csv_path):

                # Check if the simulation is in the list of desired simulations
                if desired_simulations is not None:
                    if simulation_name not in desired_simulations:
                        continue  # Skip this simulation if not desired

                # Use the DataReader to read the data from the text file into a DataFrame
                simulation_df = DataReader(sim_type).get_df_from_txt(file)

                # Insert a new column 'Simulation' with the simulation name
                simulation_df.insert(0, 'Simulation', simulation_name)

                # Append the simulation DataFrame to the dataset list

                simulation_df.to_csv(csv_path, index_label=False)

        print('Done!')

    def preprocess_simulation_data(self, row):
        """
        Extracts relevant features from a simulation data row and normalizes positions.
        """
        # Extract position coordinates and normalize X and Z
        pos = row[:, 6:9].astype(float)
        pos[:, 0] -= min(pos[:, 0])  # Normalize X
        pos[:, 2] -= min(pos[:, 2])  # Normalize Z

        # Extract n, velocity, and energy
        n = row[:, -1].astype(int)
        vel = row[:, 10:13].astype(float)
        e = row[:, 13].reshape((len(n), 1)).astype(float)

        # Combine extracted features into a single array
        x = np.concatenate((pos, vel, e, n.reshape((len(n), 1))), axis=1)

        return x

    def calculate_glass_dataset(self, train_flag=True):
        """
        Loads and processes simulation data to create a dataset for training/testing.
        """
        print('Loading edges data...')
        self.sampling_factor = 1


        data_total = []


        for file in self.path_to_data_folder.rglob('**/*data.txt'):
            print(file)
            if file.name == 'Ballon_variables_data.txt':
                self.desired_regions = {'GLASS-1.Region_1': 0, 'GLASS-1.Region_2': 0, 'LIQUID-1.Region_2': 1}
            # Extract the simulation name from the file path
            simulation_name = file.parts[-2]
            csv_path = os.path.join(os.path.dirname(file), f'{simulation_name}.csv')
            data_nodal_variables = pd.read_csv(csv_path)

            # Filter relevant regions
            df = pd.DataFrame(columns=list(data_nodal_variables.columns) + ['n'])
            if self.desired_regions != 'all':
                for reg in self.desired_regions:
                    df_region = data_nodal_variables.loc[data_nodal_variables['Region'] == reg]
                    df_region['n'] = self.desired_regions[reg]
                    df = pd.concat([df, df_region], ignore_index=True)

            # Determine the number of steps and starting step
            n_steps = (len(df['Frame_increment'].unique()) - 1)

            self.num_steps = n_steps - self.start_step
            # self.num_steps = 45 # TODO no dejar asiiiiii

            # Determine mask for downsampling (calculated ONCE per simulation for consistency)
            steps_name = df['Frame_increment'].unique()
            df_x_initial = df.loc[df['Frame_increment'] == steps_name[(0)]] 
            
            # Subsampling logic: Keep all glass, sample half of liquid
            mask = np.ones(len(df_x_initial), dtype=bool)
            
            # Robust way to find liquid particles (n=1)
            liquid_indices = np.where(df_x_initial.n == 1)[0]
            # Keep every 2nd liquid particle
            remove_liquid = liquid_indices[1::2] # Indices to remove
            mask[remove_liquid] = False
            
            # Optional: If you want random but consistent:
            # np.random.seed(42)
            # mask[n_glass:] = np.random.rand(n_liq) > 0.5

            data_sim = []
            for i in range(self.num_steps - 1 - self.sampling_dt):
                step = i + self.start_step
                df_x = df.loc[df['Frame_increment'] == steps_name[step]]
                df_y = df.loc[df['Frame_increment'] == steps_name[step + 1 + self.sampling_dt]]
                
                # Preprocess full data
                x_full = self.preprocess_simulation_data(np.asarray(df_x))
                y_full = self.preprocess_simulation_data(np.asarray(df_y))
                
                # Apply the SAME mask to both x and y to maintain particle identity
                x = torch.from_numpy(x_full[mask]).to(torch.float32)
                y = torch.from_numpy(y_full[mask]).to(torch.float32)
                
                n = x[:, -1].unsqueeze(1)
                pos = x[:, :3].clone()

                # Calculate connectivity on the downsampled set
                # Note: With fewer particles, you might need a slightly larger R_c
                newedge_index = radius_graph(pos, r=self.radius_connectivity, 
                                            loop=False, flow='source_to_target', 
                                            max_num_neighbors=1000)

                data_sim.append(Data(x[:, :-1], edge_index=newedge_index, y=y[:, :-1], n=n[:, 0]))


            # video_plot_3D(data_sim, f'output/videos/{file.stem[:-15]}.gif')
            if train_flag:
                data_sim = random.sample(data_sim, k=2 * int(len(data_sim) / 3))
                #random.shuffle(data_sim)
            data_total += data_sim

        return data_total

    def select_half_liq_particles(self, df_x, x):
        nparticulas_liq = df_x.loc[df_x.Region == 'LIQUID-1.Region_2'].shape[0]
        nparticulas = df_x.shape[0]
        nparticulas_glass = nparticulas - nparticulas_liq
        mask = np.zeros((nparticulas,), dtype=bool)
        mask[:nparticulas_glass] = True
        mask[nparticulas_glass:] = np.random.rand(nparticulas_liq) < 0.5  # True si valor < 0.5
        return x[mask]

    def calculate_glassrecons_dataset(self, train_flag=True):

        print('Loading edges data...')


        data_total = []
        for file in self.path_to_data_folder.rglob('**/*data.txt'):
            print(file)
            if file.name == 'Ballon_variables_data.txt':
                self.desired_regions = {'GLASS-1.Region_1': 0, 'GLASS-1.Region_2': 0, 'LIQUID-1.Region_2': 1}
            # Extract the simulation name from the file path
            simulation_name = file.parts[-2]
            csv_path = os.path.join(os.path.dirname(file), f'{simulation_name}.csv')
            data_nodal_variables = pd.read_csv(csv_path)

            df = pd.DataFrame(columns=list(data_nodal_variables.columns) + ['n'])
            if self.desired_regions != 'all':
                for reg in self.desired_regions:
                    df_region = data_nodal_variables.loc[data_nodal_variables['Region'] == reg]
                    df_region['n'] = self.desired_regions[reg]
                    df = pd.concat([df, df_region], ignore_index=True)

            data_sim = []
            steps_name = df['Frame_increment'].unique()
            df_x = df.loc[df['Frame_increment'] == steps_name[(self.start_step)]]

            x = torch.from_numpy(self.preprocess_simulation_data(np.asarray(df_x))).to(torch.float32)
            # x = self.select_half_liq_particles(df_x, x)  # select only the half of the particles

            y = x.clone()
            n = x[:, -1].unsqueeze(1).clone()
            x[:, 3:] = x[:, 3:] * 0
            pos = x[:, :3].clone()
            # newedge_index, _, _ = compute_connectivity(np.asarray(pos), self.radius_connectivity,
            #                                            add_self_edges=False, rnd_cnx=False)
            newedge_index = radius_graph(pos, r=self.radius_connectivity, loop=False, flow='source_to_target',
                                         max_num_neighbors=32)
            # Inicial velocity
            vel = int(file.stem[:-15].split('_')[-1]) / 1000
            velocities = torch.zeros(x.shape[0], 3)
            if file.stem[:-15].split('_')[0] == 'Glassv3':
                velocities[:, 2] = vel
            else:
                velocities[:, 0] = vel

            data_sim.append(Data(x[:, :-1], edge_index=newedge_index, y=y[:, :-1], n=n[:, 0], vel=velocities))

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection="3d")
            ax.scatter(x[:, 0], x[:, 2], x[:, 1],
                       s=32, alpha=0.8, c=y[:, 3])
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_box_aspect([1, 1, 1])
            file_path = os.path.join('images', file.name[:-19] + 'png')
            plt.savefig(file_path)
            plt.close(fig)

            data_total += data_sim
        return data_total

    def save(self, path):
        # Method to save the final dataset DataFrame as a CSV file
        self.dataset_df.to_csv(path)
        print(f'Data stored as .csv at {path}')

    def write_txt_info(self, simulation_names):
        # Save dataset information in a text file
        print('Save info dataset in txt...')

        path_txt = f'output/{self.dataset_name}_info.txt'
        with open(path_txt, 'w') as archivo:
            fecha_creacion = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            archivo.write(f"Fecha de creación: {fecha_creacion}\n\n")
            archivo.write(f"Número de simulaciones: {str(len(simulation_names))} \n")
            archivo.write(f"Separación train/val: {str(self.train_split)}/{str(1 - self.train_split)} \n")
            archivo.write(f"Connectivity radio: {str(self.radius_connectivity)} \n")
            archivo.write(f"Sampling dt: {str(self.sampling_dt)} \n")
            archivo.write(f"\n")
            archivo.write("\n".join(simulation_names))

    def save_dataset(self, data_total, simulation_names=[], train_flag=True):
        # Split the data into training and validation sets and save them as files
        if train_flag:
            data_total = random.sample(data_total, k=int(len(data_total) / 3))
            dataT_val = data_total[0:int(len(data_total) * (1 - self.train_split))]
            dataT_train = data_total[int(len(data_total) * (1 - self.train_split)):]
            torch.save(dataT_train, f'output/{self.dataset_name}_train.pt')
            torch.save(dataT_val, f'output/{self.dataset_name}_val.pt')
        else:
            torch.save(data_total, f'output/{self.dataset_name}_test.pt')

        # save info in a txt file
        self.write_txt_info(simulation_names)


def generate_pointclud(data_sim, name=''):
    t_ini = []
    for i in range(len(data_sim)):
        t_ini.append(torch.tensor(data_sim[i].x))
    z_net = torch.stack(t_ini)
    n = data_sim[0].n
    # Crear la nube de puntos inicial
    pcd = o3d.geometry.PointCloud()
    xyz = z_net[0, :, 0:3]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    data = z_net[0, :, -1]  # [n == 1]
    norm = Normalize(vmin=data.min(), vmax=data.max())
    cmap = plt.get_cmap('viridis')
    colors = cmap(norm(data))
    colors[n == 0, :] = np.array([0.8, 0.8, 0.8, 1])
    colores = o3d.utility.Vector3dVector(colors[:, :-1])
    pcd.colors = colores

    # Crear la ventana de visualización
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    view_control = visualizer.get_view_control()
    visualizer.add_geometry(pcd)
    # o3d.io.write_point_cloud(f"step_0.ply", pcd)
    # Guardar la posición inicial de la cámara
    initial_view = view_control.convert_to_pinhole_camera_parameters()

    for i in range(1, z_net.shape[0]):  # Empezamos desde 1 ya que ya hemos añadido la primera nube de puntos
        # Restaurar la posición inicial de la cámara
        view_control.convert_from_pinhole_camera_parameters(initial_view)
        view_control.rotate(i * -0.2, 80)  # Ajusta el ángulo de rotación según tus necesidades
        view_control.set_zoom(0.8)
        # view_control.rotate(0, 30)
        # Crear la nube de puntos
        pcd.points = o3d.utility.Vector3dVector(z_net[i, :, 0:3])
        data = z_net[i, :, -1]  # [n == 1]
        norm = Normalize(vmin=data.min(), vmax=data.max())
        colors = cmap(norm(data))
        colors[n == 0, :] = np.array([0.8, 0.8, 0.8, 1])
        colores = o3d.utility.Vector3dVector(colors[:, :-1])
        pcd.colors = colores
        # o3d.io.write_point_cloud(f"step_{i+1}.ply", pcd)
        # Actualizar la visualización y guardar el frame
        visualizer.update_geometry(pcd)
        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(f'images/{name}_frame_{i}.png', do_render=True)

    # Cerrar la ventana al finalizar
    visualizer.destroy_window()


def video_plot_3D(data_sim, save_dir):
    generate_pointclud(data_sim, name='gt')
    image_lst = []

    for i in range(len(data_sim) - 1):
        frame_gt = cv2.cvtColor(cv2.imread(f'images/gt_frame_{i + 1}.png'), cv2.COLOR_BGR2RGB)
        imagen_concatenada = cv2.resize(frame_gt[100:-100, 450:-450, :], None, fx=0.8, fy=0.8)
        image_lst.append(imagen_concatenada)

    imageio.mimsave(save_dir, image_lst, fps=20, loop=1)
    shutil.copy(f'images/gt_frame_{1}.png', save_dir[:-3] + 'png')
