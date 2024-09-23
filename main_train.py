import os
import json
import argparse
import datetime
import torch

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from src.dataLoader.dataset import GraphDataset
from src.gnn_nodal import NodalGNN
from src.callbacks import RolloutCallback
from src.utils.utils import str2bool


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--transfer_learning', default=False, type=str2bool, help='GPU acceleration')
    parser.add_argument('--pretrain_weights', default=r'epoch=202-val_loss=0.00.ckpt', type=str, help='name')

    # Dataset Parameters
    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dset_name', default=r'dataset_Water3D.json', type=str, help='dataset directory')

    # Save and plot options
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_dir_exp', default=r'outputs/', type=str, help='output directory')
    parser.add_argument('--experiment_name', default='exp3', type=str, help='experiment output name tensorboard')
    args = parser.parse_args()  # Parse command-line arguments

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Load dataset information from JSON file
    f = open(os.path.join(args.dset_dir, 'jsonFiles', args.dset_name))
    dInfo = json.load(f)

    # Set random seed
    pl.seed_everything(dInfo['model']['seed'], workers=True)

    # Load datasets
    train_set = GraphDataset(dInfo, os.path.join(args.dset_dir, 'datasets', dInfo['dataset']['datasetPaths']['train']))
    train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'])
    val_set = GraphDataset(dInfo, os.path.join(args.dset_dir, 'datasets', dInfo['dataset']['datasetPaths']['val']))
    val_dataloader = DataLoader(val_set, batch_size=dInfo['model']['batch_size'])
    test_set = GraphDataset(dInfo, os.path.join(args.dset_dir, 'datasets', dInfo['dataset']['datasetPaths']['test']), length=60)
    test_dataloader = DataLoader(test_set, batch_size=1)

    # Calculate scaling statistics
    scaler = train_set.get_stats()

    # Set up experiment logging
    name = f"train_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_folder = f'outputs/runs/{name}'
    wandb_logger = WandbLogger(name=name, project=dInfo['project_name'])

    # Set up callbacks
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=250, verbose=True, mode="min")
    checkpoint = ModelCheckpoint(dirpath=save_folder, filename='{epoch}-{val_loss:.2f}', monitor='val_loss',
                                 save_top_k=3)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rollout = RolloutCallback(test_dataloader)

    # Instantiate model
    nodal_gnn = NodalGNN(train_set.dims, scaler, dInfo, save_folder)
    print(nodal_gnn)
    wandb_logger.watch(nodal_gnn)

    # Load pre-trained weights if transfer learning is enabled
    if args.transfer_learning:
        path_checkpoint = os.path.join(args.dset_dir, 'weights', args.pretrain_weights)
        checkpoint_ = torch.load(path_checkpoint, map_location=device)
        nodal_gnn.load_state_dict(checkpoint_['state_dict'], strict=False)

    # Set up Trainer
    trainer = pl.Trainer(accelerator="gpu",
                         logger=wandb_logger,
                         callbacks=[checkpoint, lr_monitor, rollout, early_stop],
                         profiler="simple",
                         # gradient_clip_val=0.5,
                         num_sanity_val_steps=0,
                         max_epochs=dInfo['model']['max_epoch'],
                         deterministic=True,
                         fast_dev_run=False)

    trainer.fit(model=nodal_gnn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

