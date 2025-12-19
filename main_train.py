import os
import json
import argparse
import datetime
import torch

import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.tuner import Tuner

from src.dataLoader.dataset import GraphDataset, LMDBGraphDataset
from src.gnn_nodal import NodalGNN
from src.gnn import GNN
from src.callbacks import RolloutCallback
from src.utils.utils import str2bool

MODEL_CLASSES = {
    'GNN': GNN,
    'NodalGNN': NodalGNN,
}

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Thermodynamics-informed Graph Neural Networks')

    # Study Case
    parser.add_argument('--gpu', default=True, type=str2bool, help='GPU acceleration')
    parser.add_argument('--transfer_learning', default=False, type=str2bool, help='GPU acceleration')
    parser.add_argument('--pretrain_weights', default=r'train_NodalGNN_2025-12-17_23-52-06_epoch=99-val_loss=3.45.ckpt', type=str, help='name')
    parser.add_argument('--model', default='NodalGNN', choices=MODEL_CLASSES.keys(), help='Model to train: GNN NodalGNN')


    # Dataset Parameters
    parser.add_argument('--dset_name', default=r'dataset_Water3D.json', type=str, help='dataset directory')
 
    # Save and plot options
    parser.add_argument('--dset_dir', default='configs', type=str, help='dataset directory')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    args = parser.parse_args()  # Parse command-line arguments

    pl.seed_everything(1)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

    # Load dataset information from JSON file
    f = open(os.path.join('configs', args.dset_name))
    dInfo = json.load(f)

    # Set random seed
    pl.seed_everything(dInfo['model']['seed'], workers=True)

    train_set = GraphDataset(dInfo, os.path.join('data', 'datasets', dInfo['dataset']['datasetPaths']['train']))
    train_dataloader = DataLoader(train_set, batch_size=dInfo['model']['batch_size'], num_workers=8,  persistent_workers=True, pin_memory=False, prefetch_factor=2)
    val_set = GraphDataset(dInfo, os.path.join('data', 'datasets', dInfo['dataset']['datasetPaths']['val']))
    val_dataloader = DataLoader(val_set, batch_size=dInfo['model']['batch_size'], pin_memory=True, num_workers=2)
    test_set = GraphDataset(dInfo, os.path.join('data', 'datasets', dInfo['dataset']['datasetPaths']['test']), length=60)
    test_dataloader = DataLoader(test_set, batch_size=1)

    name = f"train_{args.model}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    save_folder = f'outputs/runs/{name}'
    wandb_logger = WandbLogger(name=name, project=dInfo['project_name'])
    wandb_logger.log_hyperparams(dInfo)

    # Set up callbacks
    early_stop = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=200, verbose=True, mode="min")
    checkpoint = ModelCheckpoint(dirpath=save_folder, filename=name'_{epoch}-{val_loss:.2f}', monitor='val_loss',
                                 save_top_k=3, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rollout = RolloutCallback(test_dataloader)

    # Instantiate model
    model_class = MODEL_CLASSES[args.model]
    model = model_class(train_set.dims, scaler, dInfo, save_folder)

    print(model)
    wandb_logger.watch(model)

    # Load pre-trained weights if transfer learning is enabled
    if args.transfer_learning:
        path_checkpoint = os.path.join(args.dset_dir, 'weights', args.pretrain_weights)
        checkpoint_ = torch.load(path_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_['state_dict'], strict=False)

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

    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(nodal_gnn, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # new_lr = lr_finder.suggestion()
    # nodal_gnn.lr = new_lr
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

