"""
Scatter plots: circuit contribution Δᵢ vs dc_power residual.

Grid layout:  rows = latent dims (z0, z1, ...)
              cols = anomaly datasets (D, S, DS)
Points coloured by residual magnitude (red = under-prediction, green = over).
Black binned-mean trend + Pearson r shown.

Usage:
    python scripts/plot_circuit_residuals.py
    python scripts/plot_circuit_residuals.py --checkpoint experiments/seed_42/best_model.pt
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import pearsonr
from scipy.stats import binned_statistic
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.models.clic import CLIC
from src.data.pv_dataset import PVDataset, compute_normalization_stats
from src.circuits.ablation import compute_circuit_contributions


# ------------------------------------------------------------------ helpers

def binned_trend(x: np.ndarray, y: np.ndarray, n_bins: int = 50) -> tuple:
    """Binned-mean trend with light Gaussian smoothing."""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    means, edges, _ = binned_statistic(x, y, statistic='mean', bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    valid = np.isfinite(means)
    centers, means = centers[valid], means[valid]
    from scipy.ndimage import gaussian_filter1d
    means = gaussian_filter1d(means, sigma=2)
    return centers, means


def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt['config']
    model = CLIC(
        main_dim        = cfg['model']['main_dim'],
        cond_dim        = cfg['model']['cond_dim'],
        seq_len         = cfg['model']['seq_len'],
        hidden_dim      = cfg['model']['hidden_dim'],
        latent_dim      = cfg['model']['latent_dim'],
        num_flows       = cfg['model']['num_flows'],
        num_lstm_layers = cfg['model']['num_lstm_layers'],
        scale_limit     = cfg['model']['scale_limit'],
        dropout         = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


def z_mean_from_train(model, cfg, stats, device, n_batches: int = 10) -> torch.Tensor:
    data_dir   = Path(cfg['data']['data_dir'])
    train_paths = [(str(data_dir / f), lbl) for f, lbl in cfg['data']['train_files']]
    ds = PVDataset(
        train_paths,
        seq_len=cfg['model']['seq_len'],
        normalize=True, stats=stats, include_labels=False,
        filter_daytime=cfg['data'].get('filter_daytime', True),
        daytime_threshold=cfg['data'].get('daytime_threshold', 0.01),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    zs = []
    with torch.no_grad():
        for i, (xm, xc) in enumerate(loader):
            if i >= n_batches:
                break
            zs.append(model.encoder(xm.to(device), xc.to(device)).cpu())
    return torch.cat(zs).mean(0)


def collect_dataset(model, cfg, stats, path_str, z_mean, device) -> dict:
    ds = PVDataset(
        [(path_str, 0)],
        seq_len=cfg['model']['seq_len'],
        normalize=True, stats=stats, include_labels=False,
        filter_daytime=cfg['data'].get('filter_daytime', True),
        daytime_threshold=cfg['data'].get('daytime_threshold', 0.01),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    residuals, deltas = [], []
    with torch.no_grad():
        for xm, xc in loader:
            xm, xc = xm.to(device), xc.to(device)
            z  = model.encoder(xm, xc)
            xr = model.decoder(z, xc)
            # dc_power residual: actual − predicted, mean over sequence (feature 0)
            res = (xm[:, 0, :] - xr[:, 0, :]).mean(dim=1)
            dlt = compute_circuit_contributions(model, xm, xc, z_mean.to(device))
            residuals.append(res.cpu().numpy())
            deltas.append(dlt.cpu().numpy())

    return {
        'residual': np.concatenate(residuals),
        'deltas':   np.concatenate(deltas, axis=0),  # [N, latent_dim]
    }


# -------------------------------------------------------------------  main

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = Path(args.checkpoint)
    output_dir = checkpoint_path.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path} ...")
    model, cfg = load_model(checkpoint_path, device)
    latent_dim = cfg['model']['latent_dim']

    data_dir   = Path(cfg['data']['data_dir'])
    train_paths = [(str(data_dir / f), lbl) for f, lbl in cfg['data']['train_files']]

    print("Computing normalization stats ...")
    stats = compute_normalization_stats(
        train_paths,
        filter_daytime=cfg['data'].get('filter_daytime', True),
        daytime_threshold=cfg['data'].get('daytime_threshold', 0.01),
    )

    print("Computing reference latent mean ...")
    z_mean = z_mean_from_train(model, cfg, stats, device)

    anomaly_datasets = [
        ('anomaly_data_no_soiling_engineered_new.csv',                 'Degradation (D)'),
        ('anomaly_data_soiling_100_no_degradation_engineered_new.csv', 'Soiling (S)'),
        ('anomaly_data_soiling_100_engineered_new.csv',                'Soiling + Degradation (DS)'),
    ]

    print("Collecting circuit contributions and residuals ...")
    collections = []
    for fname, label in anomaly_datasets:
        print(f"  {label} ...")
        col = collect_dataset(model, cfg, stats, str(data_dir / fname), z_mean, device)
        collections.append((label, col))

    # ----------------------------------------------------------------  plot

    n_cols = len(anomaly_datasets)
    col_labels = [label for label, _ in collections]

    # Shared residual range for colormap (use all datasets combined, p2–p98)
    all_res = np.concatenate([c['residual'] for _, c in collections])
    abs_max = max(abs(np.percentile(all_res, 2)), abs(np.percentile(all_res, 98)))
    norm  = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    cmap  = plt.cm.RdYlGn  # red = negative residual, green = positive

    fig, axes = plt.subplots(
        latent_dim, n_cols,
        figsize=(5.2 * n_cols, 4.0 * latent_dim),
        squeeze=False,
    )

    for col, (label, data) in enumerate(collections):
        residuals = data['residual']
        deltas    = data['deltas']

        for row in range(latent_dim):
            ax = axes[row, col]
            delta_i = deltas[:, row]

            corr, _ = pearsonr(residuals, delta_i)

            sc = ax.scatter(
                residuals, delta_i,
                c=residuals, cmap=cmap, norm=norm,
                alpha=0.25, s=4, linewidths=0, rasterized=True,
            )

            try:
                x_t, y_t = binned_trend(residuals, delta_i, n_bins=50)
                ax.plot(x_t, y_t, color='black', linewidth=1.8, zorder=5)
            except Exception:
                pass

            ax.axhline(0, color='#888888', linewidth=0.6, linestyle='--', zorder=2)
            ax.axvline(0, color='#888888', linewidth=0.6, linestyle='--', zorder=2)

            # Column header only on the top row
            if row == 0:
                ax.set_title(label, fontsize=10, pad=8)

            # Row label on the left column
            if col == 0:
                ax.set_ylabel(f'$\\Delta_{{z_{row}}}$ (norm.)', fontsize=9)
            else:
                ax.set_ylabel('')

            # x-label only on the bottom row
            if row == latent_dim - 1:
                ax.set_xlabel('dc_power residual: predicted − actual (norm.)', fontsize=9)
            else:
                ax.set_xlabel('')

            ax.tick_params(labelsize=8)

            # Panel annotation
            if row == 0:
                ann = "Primary fault circuit\n$z_0$ strongly coupled to residual"
            else:
                ann = "Secondary circuit\n$z_1$ weakly coupled to residual"
            ax.text(
                0.03, 0.97, ann,
                transform=ax.transAxes, va='top', ha='left', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.78, edgecolor='#888888', linewidth=0.6),
            )

    # Shared colourbar (right side)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('dc_power residual: predicted − actual (norm.)\n← under-prediction   over-prediction →',
                   fontsize=8)

    fig.suptitle(
        f'Circuit contributions vs DC power residual'
        f'  —  latent_dim={latent_dim}  —  {checkpoint_path.parent.name}',
        fontsize=10, y=1.01,
    )

    out_path = output_dir / 'circuit_residual_scatter.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=str,
        default='experiments/seed_42/best_model.pt',
    )
    main(parser.parse_args())
