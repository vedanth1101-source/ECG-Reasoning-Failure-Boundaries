"""
Script 09: Train Baseline Backbone and Bottleneck MLP Adapter (Phase 3)
========================================================================
Trains the 1D-ResNet baseline and Bottleneck MLP Adapter on the standardized
PTB-XL dataset (Folds 1-8 training, Fold 9 validation) for the 8 benchmark
cardiac conditions.

Pipeline:
    Input X (B, 12, 1000)
        -> ResNet1DBackbone -> h_raw (B, 512)
        -> BottleneckMLPAdapter -> Z_ECG (B, 768)
        -> TemporaryClassificationHead -> Logits (B, 8)
        -> BCEWithLogitsLoss

Saves checkpoint to: checkpoints/baseline_resnet1d_adapter.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.adapter import BottleneckMLPAdapter, TemporaryClassificationHead
from src.models.backbone_wrapper import ResNet1DBackbone
from src.models.adapted_ecg_model import AdaptedECGModel
from src.models.dataset import create_dataloader


class FullBaselineTrainingModel(nn.Module):
    """Container holding backbone, adapter, and temporary classification head for training."""

    def __init__(
        self,
        backbone_dim: int = 512,
        latent_dim: int = 768,
        num_classes: int = 8,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1DBackbone(in_channels=12, feature_dim=backbone_dim)
        self.adapter = BottleneckMLPAdapter(d_in=backbone_dim, d_mid=512, d_out=latent_dim, dropout=0.1)
        self.classifier = TemporaryClassificationHead(d_in=latent_dim, num_classes=num_classes, dropout=0.2)

        if freeze_backbone:
            self.backbone.freeze()

    def forward(self, x: torch.Tensor, lead_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h_raw = self.backbone(x, lead_mask=lead_mask)
        z_ecg = self.adapter(h_raw)
        logits = self.classifier(z_ecg)
        return z_ecg, logits

    def extract_latent(self, x: torch.Tensor, lead_mask: torch.Tensor = None) -> torch.Tensor:
        h_raw = self.backbone(x, lead_mask=lead_mask)
        return self.adapter(h_raw)


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates model computing validation loss, macro-AUC, and macro-F1."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)
            mask = batch["mask"].to(device)

            _, logits = model(x, lead_mask=mask)
            loss = criterion(logits, y)

            total_loss += loss.item() * len(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    avg_loss = total_loss / len(dataloader.dataset)

    # Compute macro AUC
    auc_scores = []
    for c in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, c])) > 1:
            auc_scores.append(roc_auc_score(all_targets[:, c], all_preds[:, c]))
    macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

    # Compute macro F1 (threshold = 0.5)
    bin_preds = (all_preds >= 0.5).astype(int)
    macro_f1 = float(f1_score(all_targets, bin_preds, average="macro", zero_division=0))

    return {
        "loss": avg_loss,
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
    }


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[*] Training on device: {device}")

    # Setup directories
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Setup dataloaders
    print("[*] Creating DataLoaders (PTB-XL Folds 1-8 train, Fold 9 val)...")
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        split="train",
        batch_size=args.batch_size,
        apply_lead_masking=args.apply_masking,
        seed=args.seed,
    )
    val_loader = create_dataloader(
        data_dir=args.data_dir,
        split="val",
        batch_size=args.batch_size,
        apply_lead_masking=False,
        seed=args.seed,
    )

    print(f"[*] Training records: {len(train_loader.dataset)}, Validation records: {len(val_loader.dataset)}")

    # Initialize model
    model = FullBaselineTrainingModel(
        backbone_dim=512,
        latent_dim=768,
        num_classes=8,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")

    # Optimizer, Loss & Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = 0.0
    best_checkpoint_path = checkpoints_dir / "baseline_resnet1d_adapter.pt"
    history = []

    print(f"[*] Commencing training for {args.epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            x = batch["signal"].to(device)
            y = batch["label"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            _, logits = model(x, lead_mask=mask)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(x)
            batch_count += 1

            if args.quick_test and batch_count >= 10:
                break

        scheduler.step()
        train_loss = running_loss / (batch_count * args.batch_size if args.quick_test else len(train_loader.dataset))

        # Validation
        val_metrics = evaluate(model, val_loader, device)

        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_macro_auc": round(val_metrics["macro_auc"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        }
        history.append(epoch_summary)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['macro_auc']:.4f} | "
            f"Val F1: {val_metrics['macro_f1']:.4f}"
        )

        # Save best model
        if val_metrics["macro_auc"] > best_val_auc:
            best_val_auc = val_metrics["macro_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "adapter_state_dict": model.adapter.state_dict(),
                    "classifier_state_dict": model.classifier.state_dict(),
                    "best_val_auc": best_val_auc,
                    "history": history,
                },
                best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint to {best_checkpoint_path} (Val AUC: {best_val_auc:.4f})")

        if args.quick_test:
            print("[*] Quick test completed successfully.")
            break

    elapsed = time.time() - start_time
    print(f"[*] Training finished in {elapsed:.1f}s. Best Val AUC: {best_val_auc:.4f}")

    # Save metrics report
    report_path = reports_dir / "baseline_training_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_val_auc": best_val_auc,
                "history": history,
                "epochs": args.epochs,
                "device": str(device),
                "total_params": total_params,
                "trainable_params": trainable_params,
            },
            f,
            indent=2,
        )
    print(f"[*] Training metrics report written to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline 1D-ResNet and Bottleneck MLP Adapter.")
    parser.add_argument("--data-dir", type=str, default="data/processed_100hz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--apply-masking", action="store_true", help="Apply random lead degradation during training.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone weights.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--quick-test", action="store_true", help="Run 1 quick epoch to verify training loop.")

    args = parser.parse_args()
    train(args)
