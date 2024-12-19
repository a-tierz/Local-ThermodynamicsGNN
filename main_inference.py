import os
import json
import argparse
import datetime
import torch
import lightning.pytorch as pl

from torch_geometric.loader import DataLoader
from src.dataLoader.dataset import GraphDataset
from src.gnn_nodal import NodalGNN
from src.utils.utils import str2bool
from src.evaluate import generate_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--pretrain_weights', default=r'pretrained_weights_beam3D.ckpt', type=str, help='name')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dset_name', default=r'dataset_Beam3D.json', type=str, help='dataset directory')

    # Save and plot options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_dir_exp', default=r'outputs/experimentes/', type=str, help='output directory')
    parser.add_argument('--experiment_name', default='exp1', type=str, help='experiment output name tensorboard')
    args = parser.parse_args()  # Parse command-line arguments

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    output_dir_exp = os.path.join(args.output_dir_exp,
                                  args.experiment_name + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # Load dataset information from JSON file
    f = open(os.path.join(args.dset_dir, 'jsonFiles', args.dset_name))
    dInfo = json.load(f)

    # Load datasets
    train_set = GraphDataset(dInfo, os.path.join(args.dset_dir, 'datasets', dInfo['dataset']['datasetPaths']['train']))
    test_set = GraphDataset(dInfo, os.path.join(args.dset_dir, 'datasets', dInfo['dataset']['datasetPaths']['test']))
    train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'])
    test_dataloader = DataLoader(test_set, batch_size=1)

    # Calculate scaling statistics
    scaler = train_set.get_stats()

    # Instantiate model
    nodal_gnn = NodalGNN(train_set.dims, scaler, dInfo, output_dir_exp)
    nodal_gnn.to(device)
    load_name = args.pretrain_weights
    load_path = os.path.join(args.dset_dir, 'weights', load_name)
    checkpoint = torch.load(load_path, map_location='cuda')
    nodal_gnn.load_state_dict(checkpoint['state_dict'])
    nodal_gnn.eval()

    # Set Trainer
    trainer = pl.Trainer(accelerator="gpu",
                         profiler="simple")

    generate_results(nodal_gnn, test_dataloader, dInfo, device, output_dir_exp, args.dset_name, args.pretrain_weights)
