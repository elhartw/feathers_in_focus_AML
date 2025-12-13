#!/usr/bin/env python3
"""
Attribute-aware bird classifier for the Feathers in Focus dataset.

Key ideas:
- ResNet backbone with two heads: class logits + attribute regression
- Image augmentation + label smoothing + optional attribute loss weighting
- Cosine LR schedule, gradient clipping, mixed precision when available
- Saves best checkpoint and writes a Kaggle-style submission.csv
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_image_path(root_dir: Path, relative: str) -> Path:
    """
    Kaggle paths start with `/train_images/123.jpg`. The extracted archive in this repo
    ended up as `root_dir/train_images/train_images/123.jpg`. We try both shapes.
    """
    rel_path = Path(relative.lstrip("/"))
    candidate = root_dir / rel_path
    if candidate.exists():
        return candidate
    nested = root_dir / rel_path.parent / rel_path
    if nested.exists():
        return nested
    raise FileNotFoundError(f"Could not resolve {relative} under {root_dir}")


class FeathersDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path,
        attributes: torch.Tensor,
        tfms: transforms.Compose,
        include_targets: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.attributes = attributes
        self.tfms = tfms
        self.include_targets = include_targets

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = resolve_image_path(self.data_root, row["image_path"])
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = self.tfms(img)

        if not self.include_targets:
            return img, int(row["id"])

        label = int(row["label"]) - 1  # zero-based for PyTorch
        attr_vec = self.attributes[label]
        return img, label, attr_vec


def build_backbone(name: str) -> Tuple[nn.Module, int]:
    name = name.lower()
    if name == "resnet34":
        backbone = models.resnet34(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feat_dim
    if name == "resnet18":
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        return backbone, feat_dim
    if name == "small_cnn":
        # Lightweight CNN for fully-from-scratch training.
        cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        return cnn, 256
    raise ValueError(f"Unsupported backbone '{name}'")


class AttributeAwareBirdNet(nn.Module):
    def __init__(
        self, num_classes: int, attr_dim: int, backbone_name: str = "small_cnn", dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.backbone, in_features = build_backbone(backbone_name)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.attr_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, attr_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        return self.classifier(feats), self.attr_head(feats)


def build_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_tfms, eval_tfms


def make_loaders(
    data_root: Path,
    attributes: torch.Tensor,
    img_size: int,
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame]:
    train_df = pd.read_csv(data_root / "train_images.csv")
    test_df = pd.read_csv(data_root / "test_images_path.csv")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(train_df, train_df["label"]))
    train_subset = train_df.iloc[train_idx].reset_index(drop=True)
    val_subset = train_df.iloc[val_idx].reset_index(drop=True)

    train_tfms, eval_tfms = build_transforms(img_size)
    train_ds = FeathersDataset(train_subset, data_root, attributes, train_tfms, include_targets=True)
    val_ds = FeathersDataset(val_subset, data_root, attributes, eval_tfms, include_targets=True)
    test_ds = FeathersDataset(test_df, data_root, attributes, eval_tfms, include_targets=False)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, test_df


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    ce_loss: nn.Module,
    attr_loss_fn: nn.Module,
    attr_weight: float,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for images, labels, attrs in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            class_logits, attr_pred = model(images)
            loss_cls = ce_loss(class_logits, labels)
            loss_attr = attr_loss_fn(attr_pred, attrs)
            loss = loss_cls + attr_weight * loss_attr

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(class_logits, labels)

    if scheduler is not None:
        scheduler.step()

    batches = len(loader)
    return running_loss / batches, running_acc / batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_loss: nn.Module,
    attr_loss_fn: nn.Module,
    attr_weight: float,
    use_amp: bool,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for images, labels, attrs in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        attrs = attrs.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            class_logits, attr_pred = model(images)
            loss_cls = ce_loss(class_logits, labels)
            loss_attr = attr_loss_fn(attr_pred, attrs)
            loss = loss_cls + attr_weight * loss_attr

        total_loss += loss.item()
        total_acc += accuracy_from_logits(class_logits, labels)

    batches = len(loader)
    return total_loss / batches, total_acc / batches


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Iterable[Tuple[int, int]]:
    model.eval()
    for images, ids in loader:
        images = images.to(device, non_blocking=True)
        logits, _ = model(images)
        preds = logits.argmax(dim=1) + 1  # back to 1-based labels
        for sample_id, pred in zip(ids, preds):
            yield int(sample_id), int(pred.item())


def save_checkpoint(path: Path, model: nn.Module, epoch: int, val_acc: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "val_acc": val_acc, "model_state": model.state_dict()}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved Feathers in Focus model")
    parser.add_argument("--data-root", type=Path, default=Path("aml-2025-feathers-in-focus"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--attr-weight", type=float, default=0.3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet34",
        choices=["resnet34", "resnet18", "small_cnn"],
        help="Model backbone trained from scratch (no pretrained weights).",
    )
    parser.add_argument("--skip-train", action="store_true", help="Only run inference using an existing checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to an existing checkpoint")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    use_amp = device.type != "cpu"

    attributes = torch.from_numpy(np.load(args.data_root / "attributes.npy")).float()
    num_classes, attr_dim = attributes.shape[0], attributes.shape[1]

    train_loader, val_loader, test_loader, test_df = make_loaders(
        data_root=args.data_root,
        attributes=attributes,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    model = AttributeAwareBirdNet(num_classes=num_classes, attr_dim=attr_dim, backbone_name=args.backbone)
    model.to(device)

    checkpoint_path = args.output_dir / "improved_bird_model.pt"
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {state.get('epoch')}, val_acc={state.get('val_acc'):.4f})")

    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    attr_loss_fn = nn.MSELoss()

    if not args.skip_train:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_acc = -1.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, device, ce_loss, attr_loss_fn, args.attr_weight, use_amp
            )
            val_loss, val_acc = evaluate(
                model, val_loader, device, ce_loss, attr_loss_fn, args.attr_weight, use_amp
            )
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(checkpoint_path, model, epoch, val_acc)
                print(f"  -> saved new best model to {checkpoint_path}")

        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state["model_state"])
            print(f"Restored best checkpoint (val_acc={state.get('val_acc'):.4f}) for inference.")

    submission_path = args.output_dir / "submission_improved.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preds = list(predict(model, test_loader, device))
    submission = pd.DataFrame(preds, columns=["id", "label"]).sort_values("id")
    submission.to_csv(submission_path, index=False)
    print(f"Wrote predictions to {submission_path}")
    print("Done.")


if __name__ == "__main__":
    main()
