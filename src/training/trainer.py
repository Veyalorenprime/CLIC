"""Two-stage training loop for CLIC.

Stage 1 — Reconstruction
    Trains encoder + decoder with reconstruction + HSIC loss.
    Flow and prior are frozen.  Runs for `epochs` epochs.

Stage 2 — Flow
    Encoder and decoder are frozen.  Trains flow + prior with NLL loss
    on the fixed latent space produced by Stage 1.
    Runs for `flow_epochs` epochs.

This eliminates gradient conflict: the flow never touches the reconstruction
path, and the encoder never has to satisfy two competing objectives at once.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

from src.losses import (
    reconstruction_loss,
    compute_hsic,
    nll_loss,
)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[Any] = None,
        save_dir: str = "experiments",
    ):
        self.device = device
        self.model  = model.to(device)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.logger       = logger

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        lr = config["training"]["learning_rate"]

        # Stage 1 optimizer: encoder + decoder only
        self.optimizer_recon = optim.Adam(
            list(model.encoder.parameters()) +
            list(model.decoder.parameters()),
            lr=lr, weight_decay=1e-5,
        )

        # Stage 2 optimizer: flow + prior only
        self.optimizer_nll = optim.Adam(
            list(model.flow.parameters()) +
            list(model.prior.parameters()),
            lr=lr, weight_decay=1e-5,
        )

        self.lambda_recon = config["loss"]["lambda_recon"]
        self.lambda_hsic  = config["loss"]["lambda_hsic"]
        self.lambda_nll   = config["loss"]["lambda_nll"]

        self.stage1_epochs = config["training"]["epochs"]
        self.stage2_epochs = config["training"].get("flow_epochs", 50)
        self.patience      = config["training"]["early_stopping_patience"]

    # ======================================================
    # Freeze helpers
    # ======================================================

    def _freeze(self, *modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(False)

    def _unfreeze(self, *modules):
        for m in modules:
            for p in m.parameters():
                p.requires_grad_(True)

    # ======================================================
    # Stage 1: one epoch of recon + HSIC
    # ======================================================

    def _train_recon_epoch(self):
        self.model.train()
        totals = {"recon": 0.0, "hsic": 0.0, "total": 0.0}
        n = 0

        pbar = tqdm(self.train_loader, desc="Stage1[recon]", leave=False)
        for x_main, x_cond in pbar:
            x_main = x_main.to(self.device)
            x_cond = x_cond.to(self.device)

            x_recon, z, _, _ = self.model(x_main, x_cond)
            l_recon = reconstruction_loss(x_recon, x_main)
            l_hsic  = compute_hsic(z, x_cond)

            if torch.isnan(l_recon) or torch.isnan(l_hsic):
                print("WARNING: NaN in recon batch, skipped.")
                continue

            loss = self.lambda_recon * l_recon + self.lambda_hsic * l_hsic

            self.optimizer_recon.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config["training"]["gradient_clip"],
            )
            self.optimizer_recon.step()

            totals["recon"] += l_recon.item()
            totals["hsic"]  += l_hsic.item()
            totals["total"] += loss.item()
            n += 1
            pbar.set_postfix(R=f"{l_recon.item():.3f}", H=f"{l_hsic.item():.3e}")

        return {k: v / max(n, 1) for k, v in totals.items()}

    def _val_recon(self):
        self.model.eval()
        totals = {"recon": 0.0, "hsic": 0.0, "total": 0.0}
        n = 0
        with torch.no_grad():
            for x_main, x_cond in self.val_loader:
                x_main = x_main.to(self.device)
                x_cond = x_cond.to(self.device)
                x_recon, z, _, _ = self.model(x_main, x_cond)
                l_recon = reconstruction_loss(x_recon, x_main)
                l_hsic  = compute_hsic(z, x_cond)
                if torch.isnan(l_recon) or torch.isnan(l_hsic):
                    continue
                loss = self.lambda_recon * l_recon + self.lambda_hsic * l_hsic
                totals["recon"] += l_recon.item()
                totals["hsic"]  += l_hsic.item()
                totals["total"] += loss.item()
                n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    # ======================================================
    # Stage 2: one epoch of NLL on frozen latent space
    # ======================================================

    def _train_nll_epoch(self):
        self.model.train()
        totals = {"nll": 0.0, "total": 0.0}
        n = 0

        pbar = tqdm(self.train_loader, desc="Stage2[nll] ", leave=False)
        for x_main, x_cond in pbar:
            x_main = x_main.to(self.device)
            x_cond = x_cond.to(self.device)

            with torch.no_grad():
                z = self.model.encoder(x_main, x_cond)

            z_flow, log_det = self.model.flow(z, x_cond)
            l_nll = nll_loss(z_flow, log_det, prior=self.model.prior, a=x_cond)

            if torch.isnan(l_nll):
                print("WARNING: NaN in NLL batch, skipped.")
                continue

            loss = self.lambda_nll * l_nll

            self.optimizer_nll.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config["training"]["gradient_clip"],
            )
            self.optimizer_nll.step()

            totals["nll"]   += l_nll.item()
            totals["total"] += loss.item()
            n += 1
            pbar.set_postfix(N=f"{l_nll.item():.3f}")

        return {k: v / max(n, 1) for k, v in totals.items()}

    def _val_nll(self):
        self.model.eval()
        totals = {"nll": 0.0, "total": 0.0}
        n = 0
        with torch.no_grad():
            for x_main, x_cond in self.val_loader:
                x_main = x_main.to(self.device)
                x_cond = x_cond.to(self.device)
                z = self.model.encoder(x_main, x_cond)
                z_flow, log_det = self.model.flow(z, x_cond)
                l_nll = nll_loss(z_flow, log_det, prior=self.model.prior, a=x_cond)
                if torch.isnan(l_nll):
                    continue
                totals["nll"]   += l_nll.item()
                totals["total"] += (self.lambda_nll * l_nll).item()
                n += 1
        return {k: v / max(n, 1) for k, v in totals.items()}

    # ======================================================
    # Main training loop
    # ======================================================

    def train(self):
        # --------------------------------------------------
        # STAGE 1: train encoder + decoder
        # --------------------------------------------------
        print("\n" + "=" * 70)
        print("STAGE 1 — Encoder + Decoder  (recon + HSIC)")
        print("=" * 70)

        self._freeze(self.model.flow, self.model.prior)
        self._unfreeze(self.model.encoder, self.model.decoder)

        best_val   = float("inf")
        patience_c = 0

        for epoch in range(self.stage1_epochs):
            train_l = self._train_recon_epoch()
            val_l   = self._val_recon()

            lr = self.optimizer_recon.param_groups[0]["lr"]
            print(
                f"\nEpoch {epoch+1}/{self.stage1_epochs} [stage1] | LR: {lr:.2e}\n"
                f"  Train -> Recon: {train_l['recon']:7.4f} | HSIC: {train_l['hsic']:7.4f}\n"
                f"  Val   -> Recon: {val_l['recon']:7.4f} | HSIC: {val_l['hsic']:7.4f}"
            )

            if val_l["total"] < best_val:
                best_val   = val_l["total"]
                patience_c = 0
                torch.save(
                    {"epoch": epoch, "model_state_dict": self.model.state_dict(),
                     "val_loss": best_val, "config": self.config},
                    self.save_dir / "best_model.pt",
                )
                print(f"  Saved best model (val={best_val:.4f})")
            else:
                patience_c += 1
                print(f"  No improvement ({patience_c}/{self.patience})")
                if patience_c >= self.patience:
                    print(f"\nEarly stopping stage 1 after {epoch+1} epochs.")
                    break

            if self.logger:
                self.logger.log({
                    "stage": 1, "epoch": epoch,
                    "train/recon": train_l["recon"], "train/hsic": train_l["hsic"],
                    "val/recon":   val_l["recon"],   "val/hsic":   val_l["hsic"],
                })

        print(f"\nStage 1 complete. Best val recon loss: {best_val:.4f}")

        # --------------------------------------------------
        # STAGE 2: train flow + prior on frozen latent space
        # --------------------------------------------------
        print("\n" + "=" * 70)
        print("STAGE 2 — Flow + Prior  (NLL on frozen latent space)")
        print("=" * 70)

        self._freeze(self.model.encoder, self.model.decoder)
        self._unfreeze(self.model.flow, self.model.prior)

        best_val   = float("inf")
        patience_c = 0

        for epoch in range(self.stage2_epochs):
            train_l = self._train_nll_epoch()
            val_l   = self._val_nll()

            lr = self.optimizer_nll.param_groups[0]["lr"]
            print(
                f"\nEpoch {epoch+1}/{self.stage2_epochs} [stage2] | LR: {lr:.2e}\n"
                f"  Train -> NLL: {train_l['nll']:7.4f}\n"
                f"  Val   -> NLL: {val_l['nll']:7.4f}"
            )

            if val_l["total"] < best_val:
                best_val   = val_l["total"]
                patience_c = 0
                torch.save(
                    {"epoch": epoch, "model_state_dict": self.model.state_dict(),
                     "val_loss": best_val, "config": self.config},
                    self.save_dir / "best_model.pt",
                )
                print(f"  Saved best model (val NLL={best_val:.4f})")
            else:
                patience_c += 1
                print(f"  No improvement ({patience_c}/{self.patience})")
                if patience_c >= self.patience:
                    print(f"\nEarly stopping stage 2 after {epoch+1} epochs.")
                    break

            if self.logger:
                self.logger.log({
                    "stage": 2, "epoch": epoch,
                    "train/nll": train_l["nll"],
                    "val/nll":   val_l["nll"],
                })

        print(f"\nStage 2 complete. Best val NLL: {best_val:.4f}")

        # Save final model
        print("\n" + "=" * 70)
        print("SAVING FINAL MODEL")
        print("=" * 70)
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict(),
             "val_loss": best_val, "config": self.config},
            self.save_dir / "final_model.pt",
        )
        print(f"Final model saved to: {self.save_dir / 'final_model.pt'}")
        print("=" * 70)
