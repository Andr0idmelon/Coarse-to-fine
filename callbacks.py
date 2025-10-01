import os, weakref, torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt

class TrackableModelCheckpoint(ModelCheckpoint):
    """Extended ModelCheckpoint to track best model path"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_model_path = ""
        self.last_model_path = ""
        self.trainer_ref = None
    
    def on_train_start(self, trainer, pl_module):
        """Save trainer weak reference"""
        super().on_train_start(trainer, pl_module)
        self.trainer_ref = weakref.ref(trainer)
    
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        """Record saved paths"""
        # Call parent method first
        result = super().on_save_checkpoint(trainer, pl_module, checkpoint)
        
        # Update last saved model path
        if hasattr(self, '_last_checkpoint_saved') and self._last_checkpoint_saved:
            self.last_model_path = self._last_checkpoint_saved
            
            # If current is best model, update best model path
            if hasattr(self, 'best_k_models') and self.best_k_models:
                best_score = min(self.best_k_models.values()) if self.mode == "min" else max(self.best_k_models.values())
                for path, score in self.best_k_models.items():
                    if score == best_score:
                        self.best_model_path = path
                        break

        return result
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Update best model path at validation epoch end"""
        super().on_validation_epoch_end(trainer, pl_module)
        
        # No additional operations here to prevent API incompatibility
        # Best model path will be updated in on_save_checkpoint

# Custom visualization callback
class VisualizationCallback(pl.Callback):
    """
    Visualize prediction results for the first batch at the end of each validation epoch.
    """
    def __init__(self, output_dir, run_name, vis_interval=10):
        super().__init__()
        self.save_dir = os.path.join(output_dir, "visualizations", run_name)
        self.vis_interval = vis_interval
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Visualization callback initialized. Images will be saved to: {self.save_dir}")

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # Only visualize on main process & at specified intervals
        if epoch % self.vis_interval != 0: return
        if dist.is_initialized() and dist.get_rank() != 0: return

        if not hasattr(pl_module, 'val_vis_batch'): 
            return

        (gray, binary), gt = pl_module.val_vis_batch
        pred = pl_module.val_vis_outputs['final_flow'][0].detach().cpu().numpy()
        mag = np.sqrt(pred[0]**2 + pred[1]**2)

        fig, ax = plt.subplots()
        im = ax.imshow(mag, cmap='viridis')
        fig.colorbar(im, ax=ax)
        fig.suptitle(f'Epoch {epoch}')
        fig.savefig(os.path.join(self.save_dir, f'val_flow_{epoch}.png'))
        plt.close(fig)

def set_callbacks(params):
    """Setup training callbacks based on configuration parameters"""
    callbacks = []
    
    # Early stopping callback
    if params.training.early_stop == True:
        early_stopping = EarlyStopping(
            monitor=params.training.early_stop_settings.monitor,
            patience=params.training.early_stop_settings.patience,
            verbose=params.training.early_stop_settings.verbose
        )
        callbacks.append(early_stopping)
    
    # Create model directory
    model_dir = os.path.join(params.training.output_dir, 'models', params.training.wandb.name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save best model checkpoint
    best_checkpoint = TrackableModelCheckpoint(
        dirpath=model_dir,
        monitor='val_loss',
        save_top_k=1,  # Only save the best 1 model
        filename='best_{epoch:03d}_{val_loss:.4f}',
        mode='min',
        save_last=False  # Don't save last model here
    )
    callbacks.append(best_checkpoint)
    
    # Save last model checkpoint separately
    last_checkpoint = TrackableModelCheckpoint(
        dirpath=model_dir,
        filename='last',
        save_last=True,
        save_top_k=0,  # Don't save based on metrics
        save_weights_only=False,
        every_n_epochs=1  # Save every epoch
    )
    callbacks.append(last_checkpoint)

    # Create visualization directory
    vis_dir = os.path.join(params.training.output_dir, 'visualizations', params.training.wandb.name)
    os.makedirs(vis_dir, exist_ok=True)
    
    return callbacks, best_checkpoint, last_checkpoint
