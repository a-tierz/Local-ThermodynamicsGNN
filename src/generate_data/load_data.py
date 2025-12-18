from builder import GraphDataset
import argparse
import os

Type_proj = ['Glass', 'Glass_recons']
selc = 0
if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Abaqus Geometry Modifier')
    parser.add_argument('--path_to_data', type=str, default=rf'data')
    parser.add_argument('--save_csv', type=bool, default=False)
    parser.add_argument('--train', type=bool, default=False)
    args = parser.parse_args()  # Parse command-line arguments

    if args.train:
        print('\nPreparing TRAIN dataset...')
        folder_path = os.path.join(args.path_to_data, Type_proj[selc], 'train')
        train_graph_dataset = GraphDataset(folder_path, sim_type=Type_proj[selc],
                                           simulation_names=os.listdir(folder_path),
                                           dataset_name='V150_Glass_Downsampled_rc012_train', train_flag=True, save_csv=args.save_csv,
                                           desired_regions={'GLASS-1.Region_1': 0, 'GLASS-1.Region_4': 0, 'LIQUID-1.Region_2': 1})
    else:
        folder_path = os.path.join(args.path_to_data, Type_proj[selc], 'test')
        for name_test in os.listdir(folder_path):

            train_graph_dataset = GraphDataset(os.path.join(folder_path, name_test), sim_type=Type_proj[selc],
                                               simulation_names=[name_test],
                                               dataset_name='V150_Glass_Downsampled_rc012_' + name_test, train_flag=False, save_csv=args.save_csv,
                                               desired_regions={'GLASS-1.Region_1': 0, 'GLASS-1.Region_4': 0, 'LIQUID-1.Region_2': 1})



