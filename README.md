# CLIC — Causal Latent Circuits for Interpretable OOD Detection

CLIC is an unsupervised fault-detection framework for photovoltaic (PV) systems. It learns normal PV dynamics from clean data and detects anomalies at inference time by decomposing output deviations into per-latent **circuit contributions** and a flow-based **OOD score**.

---

## How It Works

CLIC uses a two-stage causal model:

1. **Conditional LSTM encoder** `g(x, a) → z` — compresses electrical measurements into a low-dimensional latent code, conditioned on environmental features.
2. **Additive LSTM decoder** `h∅(a) + Σᵢ fᵢ(zᵢ, a) → x̂` — reconstructs observations as a sum of per-circuit contributions, each driven by one latent dimension.
3. **Conditional normalizing flow** `ϕ_a(z) → z′` — maps the latent code to a base space where `p(z′|a)` is Gaussian. Anomaly detection uses `-log p(z′|a)`.

An **HSIC penalty** enforces that `z` is statistically independent of `a`, ensuring the latent encodes the panel's internal state rather than recoding the environment.

**Two-stage training:**
- Stage 1 — encoder + decoder with reconstruction + HSIC loss (flow frozen).
- Stage 2 — flow + prior with NLL on the fixed latent space (encoder/decoder frozen).

---

## Project Structure

```
├── configs/
│   └── base_config.yaml          # Model, training, and data hyperparameters
│
├── src/
│   ├── models/
│   │   ├── clic.py               # Full model composition
│   │   ├── encoder.py            # Conditional LSTM encoder
│   │   ├── decoder.py            # Additive LSTM decoder
│   │   └── flow.py               # Conditional RealNVP
│   ├── data/
│   │   ├── pv_dataset.py         # PVDataset + normalization
│   │   └── preprocessing.py      # Train/val splits
│   ├── losses/
│   │   ├── reconstruction.py     # MSE loss
│   │   ├── hsic.py               # HSIC independence penalty
│   │   └── flow_nll.py           # NLL + conditional prior
│   ├── circuits/
│   │   ├── ablation.py           # Mean-ablation circuit contributions Δᵢ
│   │   └── ood_score.py          # OOD score −log p(z′|a)
│   ├── training/
│   │   ├── trainer.py            # Two-stage training loop
│   │   └── logger.py             # W&B integration
│   └── utils/
│       ├── config.py
│       └── seed.py
│
├── scripts/
│   ├── prepare_pv_data.py        # Data preprocessing
│   ├── train.py                  # Training entry point
│   ├── plot_circuit_summary.py   # Experiment 1: |Δᵢ| bars + OOD score
│   ├── plot_circuit_residuals.py # Experiment 1: circuit contribution scatter
│   └── compare_linear_baseline.py # Experiment 2: causal vs. correlation ROC-AUC
│
└── tests/
    ├── test_models.py
    ├── test_losses.py
    └── test_circuits.py
```


## Training

```bash
python scripts/train.py --config configs/base_config.yaml
# optional overrides:
python scripts/train.py --config configs/base_config.yaml --seed 123 --save_dir experiments/run1
```

Checkpoints are saved to `experiments/<save_dir>/best_model.pt`.

---

## Experiments

### Experiment 1 — Circuit Analysis

Visualises circuit contributions `|Δᵢ|` and OOD scores across fault modes (Normal / Degradation / Soiling / DS), and plots per-circuit contribution vs. dc_power residual.

```bash
python scripts/plot_circuit_summary.py  --checkpoint experiments/test/best_model.pt
python scripts/plot_circuit_residuals.py --checkpoint experiments/test/best_model.pt
```

Outputs: `figures/circuit_summary.png`, `figures/circuit_residual_scatter.png`

### Experiment 2 — Causal vs. Correlation-Based Detection

Compares CLIC's causal signals (OOD score, residual) against a linear regression baseline (`x_cond → dc_power`) on ROC-AUC across all fault types. Inspired by Langdon & Engel (2025, *Nature Neuroscience*).

```bash
python scripts/compare_linear_baseline.py --checkpoint experiments/test/best_model.pt
```

Output: `figures/linear_vs_clic_comparison.png`

| Method | AUC (D) | AUC (S) | AUC (DS) |
|---|---|---|---|
| Linear residual | 0.594 | 0.596 | 0.604 |
| CLIC residual | 0.505 | 0.504 | 0.504 |
| **CLIC OOD** | **0.764** | **0.763** | **0.762** |
| CLIC combined (α=0.5) | 0.660 | 0.655 | 0.659 |

---

## Tests

```bash
pytest tests/ -v        # 13 tests
```

---

## Key Hyperparameters (`configs/base_config.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `latent_dim` | 2 | Number of circuit dimensions |
| `num_flows` | 4 | RealNVP coupling layers |
| `lambda_hsic` | 0.01 | Independence penalty weight |
| `lambda_nll` | 0.1 | Flow NLL weight |
| `epochs` | 100 | Stage-1 epochs |
| `flow_epochs` | 50 | Stage-2 epochs |
