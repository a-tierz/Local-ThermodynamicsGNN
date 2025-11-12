import os
import json
import argparse
import datetime
import torch
import lightning.pytorch as pl

from torch_geometric.loader import DataLoader
from src.dataLoader.dataset import GraphDataset
from src.gnn_nodal import NodalGNN
from src.gnn import GNN
from src.utils.utils import str2bool
from src.evaluate import generate_results, generate_results_recons

MODEL_CLASSES = {
    'GNN': GNN,
    'NodalGNN': NodalGNN,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--pretrain_weights', default=r'epoch=499-val_loss=0.00.ckpt', type=str, help='name')
    parser.add_argument('--model', default='GNN', choices=MODEL_CLASSES.keys(), help='Model to train: GNN NodalGNN')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dset_name', default=r'dataset_Water3D.json', type=str, help='dataset directory')

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
    model_class = MODEL_CLASSES[args.model]

    path_checkpoint = os.path.join(args.dset_dir, 'weights', args.pretrain_weights)
    model = model_class.load_from_checkpoint(path_checkpoint, dt_info=dInfo)
    model.to(device)
    model.eval()

    # Set Trainer
    trainer = pl.Trainer(accelerator="gpu",
                         profiler="simple")

    # generate_results(nodal_gnn, test_dataloader, dInfo, device, output_dir_exp, args.dset_name, args.pretrain_weights)
    generate_results_recons(model, test_dataloader, dInfo, device, output_dir_exp, args.dset_name, args.pretrain_weights)
