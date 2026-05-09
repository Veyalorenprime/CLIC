"""Experiment tracking and logging"""

from typing import Dict, Any, Optional


class WandbLogger:
    """Weights & Biases logger wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        try:
            import wandb
            self.wandb = wandb
            
            wandb_config = config.get('logging', {})
            project = wandb_config.get('wandb_project', 'clic')
            entity = wandb_config.get('wandb_entity')
            
            wandb.init(
                project=project,
                entity=entity,
                config=config,
                mode='online' if wandb_config.get('use_wandb', False) else 'disabled'
            )
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            self.wandb = None
    
    def log(self, metrics: Dict[str, Any]):
        """Log metrics."""
        if self.wandb is not None:
            self.wandb.log(metrics)
    
    def finish(self):
        """Finish logging."""
        if self.wandb is not None:
            self.wandb.finish()
