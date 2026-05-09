"""
Main training script for CLIC

Usage:
    python scripts/train.py --config configs/base_config.yaml
    python scripts/train.py --config configs/base_config.yaml --seed 123 --gpu 0
    python scripts/train.py --config configs/base_config.yaml --save_dir experiments/my_run
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import argparse
import yaml
import torch
from pathlib import Path

from src.utils.seed import seed_everything
from src.utils.config import load_config
from src.data.pv_dataset import get_dataloaders
from src.models.clic import CLIC
from src.training.trainer import Trainer
from src.training.logger import WandbLogger


def validate_config(config):
    """Validate config and auto-fix seq_len inconsistency."""
    if config['data']['seq_len'] != config['model']['seq_len']:
        config['data']['seq_len'] = config['model']['seq_len']

    for param in ['main_dim', 'cond_dim', 'seq_len', 'hidden_dim', 'latent_dim',
                  'num_flows', 'num_lstm_layers', 'scale_limit', 'dropout']:
        if param not in config['model']:
            raise ValueError(f"Missing model parameter: {param}")

    for param in ['lambda_recon', 'lambda_hsic', 'lambda_nll']:
        if param not in config['loss']:
            raise ValueError(f"Missing loss parameter: {param}")

    for param in ['batch_size', 'learning_rate', 'weight_decay',
                  'gradient_clip', 'epochs', 'early_stopping_patience']:
        if param not in config['training']:
            raise ValueError(f"Missing training parameter: {param}")

    for param in ['data_dir', 'seq_len', 'filter_daytime',
                  'daytime_threshold', 'train_files', 'val_files']:
        if param not in config['data']:
            raise ValueError(f"Missing data parameter: {param}")

    return config


def print_config_summary(config):
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print("\n[MODEL]")
    for k, v in config['model'].items():
        print(f"  {k:20s}: {v}")
    print("\n[LOSS]")
    for k, v in config['loss'].items():
        print(f"  {k:20s}: {v}")
    print("\n[TRAINING]")
    for k, v in config['training'].items():
        print(f"  {k:20s}: {v}")
    print("\n[DATA]")
    print(f"  data_dir            : {config['data']['data_dir']}")
    print(f"  seq_len             : {config['data']['seq_len']}")
    print(f"  filter_daytime      : {config['data']['filter_daytime']}")
    print(f"  num_train_files     : {len(config['data']['train_files'])}")
    print(f"  num_val_files       : {len(config['data']['val_files'])}")
    print("=" * 70 + "\n")


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Override with command-line args
    if args.seed is not None:
        config['seed'] = args.seed
    if args.gpu is not None:
        config['gpu'] = args.gpu
    
    config = validate_config(config)
    print_config_summary(config)
    
    # Handle save_dir override
    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
    else:
        # Default: use config's save_dir + seed
        save_dir = Path(config['logging']['save_dir']) / f"seed_{config['seed']}"
    
    # Seed everything for reproducibility
    seed_everything(config['seed'])
    
    # Setup device
    device = torch.device(f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    print(f"Saving to: {save_dir}\n")
    
    # Initialize logger
    if config['logging']['use_wandb']:
        logger = WandbLogger(config)
    else:
        logger = None
    
    train_loader, val_loader = get_dataloaders(config)

    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = CLIC(
        main_dim=config['model']['main_dim'],
        cond_dim=config['model']['cond_dim'],
        seq_len=config['model']['seq_len'],
        hidden_dim=config['model']['hidden_dim'],
        latent_dim=config['model']['latent_dim'],
        num_flows=config['model']['num_flows'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        scale_limit=config['model']['scale_limit'],
        dropout=config['model']['dropout'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        logger=logger,
        save_dir=str(save_dir)
    )
    
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")

    trainer.train()
    
    if logger:
        logger.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIC model")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (overrides config)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU device ID (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Save directory (overrides config)')
    
    args = parser.parse_args()
    
    main(args)