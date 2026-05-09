"""
Experiment 2: Causal (CLIC) vs. correlation-based (linear regression) comparison.

Inspired by Langdon & Engel (2025) Fig. 7: correlation-based methods conflate
fault signal with environmental variability, while causal circuit models separate
them. Here we compare:

  1. Linear baseline  — LinearRegression from x_cond -> dc_power (trained on normal data)
  2. CLIC residual    — actual - CLIC decoder prediction
  3. CLIC OOD         — -log p(z'|a)
  4. CLIC combined    — alpha * s_OOD_z + (1-alpha) * residual_z

ROC-AUC is computed for Normal vs. D, Normal vs. S, Normal vs. DS.
Output: ROC curves + AUC bar chart.

Usage:
    python scripts/compare_linear_baseline.py
    python scripts/compare_linear_baseline.py --checkpoint experiments/test/best_model.pt --alpha 0.5
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from src.models.clic import CLIC
from src.data.pv_dataset import PVDataset, compute_normalization_stats
from src.circuits.ood_score import compute_ood_score


# ------------------------------------------------------------------ helpers

def load_model(checkpoint_path: Path, device: torch.device) -> tuple:
    ckpt  = torch.load(checkpoint_path, map_location=device)
    cfg   = ckpt['config']
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


def collect(model, cfg, stats, path_str, device) -> dict:
    ds = PVDataset(
        [(path_str, 0)],
        seq_len=cfg['model']['seq_len'],
        normalize=True, stats=stats, include_labels=False,
        filter_daytime=cfg['data'].get('filter_daytime', True),
        daytime_threshold=cfg['data'].get('daytime_threshold', 0.01),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    dc_actual, dc_clic_pred, x_conds, oods = [], [], [], []
    with torch.no_grad():
        for xm, xc in loader:
            xm, xc = xm.to(device), xc.to(device)
            z  = model.encoder(xm, xc)
            xr = model.decoder(z, xc)
            s  = compute_ood_score(model, xm, xc)

            dc_actual.append(xm[:, 0, :].mean(1).cpu().numpy())
            dc_clic_pred.append(xr[:, 0, :].mean(1).cpu().numpy())
            x_conds.append(xc.cpu().numpy())
            oods.append(s.cpu().numpy())

    return {
        'dc_actual':    np.concatenate(dc_actual),
        'dc_clic_pred': np.concatenate(dc_clic_pred),
        'x_cond':       np.concatenate(x_conds),
        'ood':          np.concatenate(oods),
    }


def fit_linear_baseline(cfg, stats, train_paths, device) -> LinearRegression:
    """Fit x_cond -> dc_power_mean on normal training data (no model needed)."""
    ds = PVDataset(
        train_paths,
        seq_len=cfg['model']['seq_len'],
        normalize=True, stats=stats, include_labels=False,
        filter_daytime=cfg['data'].get('filter_daytime', True),
        daytime_threshold=cfg['data'].get('daytime_threshold', 0.01),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    X, y = [], []
    for xm, xc in loader:
        X.append(xc.numpy())
        y.append(xm[:, 0, :].mean(1).numpy())

    X = np.concatenate(X)
    y = np.concatenate(y)
    reg = LinearRegression().fit(X, y)
    print(f"  Linear baseline R² on train: {reg.score(X, y):.4f}")
    return reg


def compute_scores(data: dict, reg: LinearRegression,
                   mu_ood, sig_ood, mu_res, sig_res, alpha) -> dict:
    """Return per-sample anomaly scores for all four methods.

    Convention: higher score = more anomalous.
    For residual methods: predicted - actual (positive when panel underperforms).
    """
    linear_pred  = reg.predict(data['x_cond'])
    linear_score = linear_pred - data['dc_actual']
    clic_score   = data['dc_clic_pred'] - data['dc_actual']
    ood_score    = data['ood']

    ood_z    = (ood_score  - mu_ood) / (sig_ood + 1e-8)
    res_z    = (clic_score - mu_res) / (sig_res + 1e-8)
    combined = alpha * ood_z + (1 - alpha) * res_z

    return {
        'Linear residual':            linear_score,
        'CLIC residual':              clic_score,
        'CLIC OOD':                   ood_score,
        f'CLIC combined (a={alpha})': combined,
    }


def safe_auc(y_true, y_scores) -> tuple:
    """ROC-AUC, always >= 0.5 (flip score sign if needed). Returns (auc, flipped)."""
    auc = roc_auc_score(y_true, y_scores)
    if auc < 0.5:
        return 1.0 - auc, True
    return auc, False


# -------------------------------------------------------------------  main

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha  = args.alpha

    checkpoint_path = Path(args.checkpoint)
    output_dir      = checkpoint_path.parent / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint_path} ...")
    model, cfg  = load_model(checkpoint_path, device)
    data_dir    = Path(cfg['data']['data_dir'])
    train_paths = [(str(data_dir / f), lbl) for f, lbl in cfg['data']['train_files']]

    print("Computing normalization stats ...")
    stats = compute_normalization_stats(
        train_paths,
        filter_daytime=cfg['data'].get('filter_daytime', True),
        daytime_threshold=cfg['data'].get('daytime_threshold', 0.01),
    )

    print("Fitting linear baseline ...")
    reg = fit_linear_baseline(cfg, stats, train_paths, device)

    datasets = [
        ('Normal', str(data_dir / 'processed/splits/normal_val.csv')),
        ('D',      str(data_dir / 'anomaly_data_no_soiling_engineered_new.csv')),
        ('S',      str(data_dir / 'anomaly_data_soiling_100_no_degradation_engineered_new.csv')),
        ('DS',     str(data_dir / 'anomaly_data_soiling_100_engineered_new.csv')),
    ]

    print("Collecting signals ...")
    raw = {}
    for mode, path_str in datasets:
        print(f"  {mode} ...")
        raw[mode] = collect(model, cfg, stats, path_str, device)

    # Normalisation parameters from Normal dataset (for combined signal)
    normal_clic_res = raw['Normal']['dc_clic_pred'] - raw['Normal']['dc_actual']
    mu_ood  = raw['Normal']['ood'].mean()
    sig_ood = raw['Normal']['ood'].std()
    mu_res  = normal_clic_res.mean()
    sig_res = normal_clic_res.std()

    all_scores = {}
    for mode in raw:
        all_scores[mode] = compute_scores(
            raw[mode], reg, mu_ood, sig_ood, mu_res, sig_res, alpha,
        )

    method_names  = list(all_scores['Normal'].keys())
    anomaly_modes = ['D', 'S', 'DS']
    anomaly_labels = [
        'Degradation (D)',
        'Soiling (S)',
        'Soiling + Degradation (DS)',
    ]

    # ---- AUC table ----
    auc_table = {}
    for method in method_names:
        auc_table[method] = {}
        for mode in anomaly_modes:
            y_true   = np.concatenate([
                np.zeros(len(all_scores['Normal'][method])),
                np.ones(len(all_scores[mode][method])),
            ])
            y_scores = np.concatenate([
                all_scores['Normal'][method],
                all_scores[mode][method],
            ])
            auc, _ = safe_auc(y_true, y_scores)
            auc_table[method][mode] = auc

    print(f"\n{'Method':<38}  {'D':>7}  {'S':>7}  {'DS':>7}")
    print('-' * 62)
    for method in method_names:
        row = f'{method:<38}'
        for mode in anomaly_modes:
            row += f"  {auc_table[method][mode]:>7.4f}"
        print(row)

    # ---- Plot ----
    colours   = ['#888888', '#4878CF', '#D65F5F', '#E8A838']
    n_methods = len(method_names)
    n_anomaly = len(anomaly_modes)

    fig, axes = plt.subplots(1, n_anomaly + 1,
                             figsize=(5.2 * (n_anomaly + 1), 4.5))

    # ROC curves
    for col, (mode, label) in enumerate(zip(anomaly_modes, anomaly_labels)):
        ax = axes[col]
        for method, c in zip(method_names, colours):
            y_true   = np.concatenate([
                np.zeros(len(all_scores['Normal'][method])),
                np.ones(len(all_scores[mode][method])),
            ])
            y_scores = np.concatenate([
                all_scores['Normal'][method],
                all_scores[mode][method],
            ])
            auc, flipped = safe_auc(y_true, y_scores)
            if flipped:
                y_scores = -y_scores
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            ax.plot(fpr, tpr, color=c, linewidth=1.6,
                    label=f'{method}  ({auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.7, alpha=0.5)
        ax.set_title(label, fontsize=10, pad=6)
        ax.set_xlabel('False positive rate', fontsize=9)
        if col == 0:
            ax.set_ylabel('True positive rate', fontsize=9)
        ax.legend(fontsize=7, loc='lower right')
        ax.tick_params(labelsize=8)

    # AUC bar chart
    ax = axes[-1]
    x  = np.arange(n_anomaly)
    w  = 0.75 / n_methods
    for i, (method, c) in enumerate(zip(method_names, colours)):
        aucs = [auc_table[method][m] for m in anomaly_modes]
        ax.bar(x + i * w - 0.375 + w / 2, aucs,
               width=w, color=c, label=method, alpha=0.85, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(['D', 'S', 'DS'], fontsize=9)
    ax.set_ylabel('ROC-AUC', fontsize=9)
    ax.set_ylim(0.45, 1.02)
    ax.axhline(0.5, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
    ax.set_title('AUC summary', fontsize=10, pad=6)
    ax.legend(fontsize=7, loc='lower right')
    ax.tick_params(labelsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    fig.suptitle(
        f'Causal vs. correlation-based anomaly detection  —  {checkpoint_path.parent.name}',
        fontsize=10, y=1.01,
    )
    fig.tight_layout()

    out_path = output_dir / 'linear_vs_clic_comparison.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/test/best_model.pt')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for OOD in combined signal')
    main(parser.parse_args())
