import os
import lightning.pytorch as pl
from pathlib import Path
import wandb
import shutil
from src.utils.plots import plot_2D, plot_3D, plot_2D_image, plot_image3D, plotError, plot_1D, video_plot_3D
from src.evaluate import roll_out,compute_error, print_error


class RolloutCallback(pl.Callback):
    def __init__(self, dataloader, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = dataloader

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins."""
        pl_module.rollouts_z_t1_pred = []
        pl_module.rollouts_z_t1_gt = []
        pl_module.rollouts_idx = []

        folder_path = Path(trainer.checkpoint_callback.dirpath) / 'videos'
        if folder_path.exists():
            # Remove the folder and its contents
            shutil.rmtree(folder_path)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch > 0 and trainer.current_epoch%pl_module.rollout_freq == 0:
            try:
                z_net, z_gt, t = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.dtset_type)

                save_dir = os.path.join(pl_module.save_folder, f'epoch_{trainer.current_epoch}.gif')
                # plot_1D(z_net, z_gt, q_0, save_dir=save_dir, var=0)
                if pl_module.data_dim == 2:
                    plot_2D(z_net[:t, :, :], z_gt[:t, :, :], save_dir=save_dir, var=5)
                else:
                    # plot_3D(z_net[:t, :, :], z_gt[:t, :, :], save_dir=save_dir, var=5)
                    video_plot_3D(z_net[:t, :, :], z_gt[:t, :, :], save_dir=save_dir)
                trainer.logger.experiment.log({"rollout": wandb.Video(save_dir, format='gif')})
            except:
                print('The rollout has failed')

    def on_train_end(self, trainer, pl_module):
        z_net, z_gt, q_0 = roll_out(pl_module, self.dataloader, pl_module.device, pl_module.radius_connectivity, pl_module.dtset_type)
        filePath = os.path.join(pl_module.save_folder, 'metrics.txt')
        save_dir = os.path.join(pl_module.save_folder, f'final_{trainer.current_epoch}.gif')
        with open(filePath, 'w') as f:
            error, L2_list = compute_error(z_net, z_gt,pl_module.state_variables)
            lines = print_error(error)
            f.write('\n'.join(lines))
            print("[Test Evaluation Finished]\n")
            f.close()
        plotError(z_gt, z_net, L2_list, pl_module.state_variables, pl_module.data_dim, pl_module.save_folder)
        if pl_module.data_dim == 2:
            plot_2D(z_net, z_gt, save_dir=save_dir, var=5)
            plot_2D_image(z_net, z_gt, -1, var=5, output_dir=pl_module.save_folder)
        else:
            plot_3D(z_net, z_gt, save_dir=save_dir, var=5)
            data = [sample for sample in self.dataloader]
            plot_image3D(z_net, z_gt, pl_module.save_folder, var=5, step=-1, n=data[0].n)



