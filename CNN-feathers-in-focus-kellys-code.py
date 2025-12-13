
# IMPORTS
from typing import Tuple, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from torch.optim.lr_scheduler import StepLR


# DATA SETUP
EPOCHS = 25
ATTR_WEIGHT = 0.5

DATA_DIR = Path("aml-2025-feathers-in-focus")
TRANSFORM_TRAIN = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.7, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(10),
    v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    v2.RandomErasing(p=0.25),
])

TRANSFORM_VAL = v2.Compose([
    v2.Resize(int(224 * 1.14)),
    v2.CenterCrop(224),
    v2.ToTensor(),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load attributes
attributes = np.load(DATA_DIR / "attributes.npy")
attributes = torch.tensor(attributes, dtype=torch.float32)
print(f"Attributes shape: {attributes.shape}")


# DATASET
class BirdDataset(Dataset):
    def __init__(
            self, 
            df, 
            search_root, 
            transform, 
            attributes, 
            include_targets: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.search_root = search_root
        self.transform = transform
        self.attributes = attributes
        self.include_targets = include_targets
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = Path(str(row["image_path"])).name
        path = self.search_root / filename.lstrip('/')
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        
        if not self.include_targets:
            return img, int(row["id"])
        
        label = int(row["label"]) - 1
        attrs = self.attributes[label]
        return image, label, attrs


# DATA LOADERS
def data_loaders(
        search_root: Path,
        attributes: torch.Tensor,
        img_size: int,
        batch_size: int,
        val_split: float,
        seed: int,
        num_workers: int,
):
    train_df = pd.read_csv(search_root / "train_images.csv")
    test_df = pd.read_csv(search_root / "test_images_path.csv")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_split, random_state=seed
    )
    
    train_idx, val_idx = next(splitter.split(train_df, train_df["label"]))

    train_dataset = BirdDataset(
        df=train_df.iloc[train_idx].reset_index(drop=True),
        search_root=DATA_DIR / "train_images",
        transform=TRANSFORM_TRAIN,
        attributes=attributes,
    )
    val_dataset = BirdDataset(
        df=train_df.iloc[val_idx].reset_index(drop=True),
        search_root=DATA_DIR / "train_images",
        transform=TRANSFORM_VAL,
        attributes=attributes,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers= num_workers, pin_memory= True, persistent_workers= num_workers > 0,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers= num_workers, pin_memory= True, persistent_workers= num_workers > 0,shuffle=False)
    test_loader = DataLoader(test_df, batch_size=batch_size, num_workers= num_workers, pin_memory= True, persistent_workers= num_workers > 0,shuffle=False)
    return train_loader, val_loader, test_loader, test_df

# MODEL
class Net(nn.Module):
    def __init__(self, num_classes: int, attr_dim: int, dropout: float = 0.2) -> None:
        super().__init__()

        self.backbone = nn.Sequential(
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
    
        in_features = 256

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
    
    def forward(self, x):
        # Feature extraction
        features = self.backbone(x)
        
        # Classification output
        class_out = self.classifier(features)
        
        # Attribute output
        attr_out = self.attr_head(features)
        
        return class_out, attr_out


# TRAINING
def train(
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
        preds = class_logits.argmax(dim=1)
        running_acc += (preds == labels).float().mean().item()

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type != "cpu"
print(f"Device: {device}")

attributes = torch.from_numpy(np.load("aml-2025-feathers-in-focus/attributes.npy")).float()
num_classes, attr_dim = attributes.shape[0], attributes.shape[1]
search_root = DATA_DIR

train_loader, val_loader, test_loader, test_df = data_loaders(
    search_root=search_root,
    attributes=attributes,
    img_size=224,
    batch_size=32,
    val_split=0.15,
    seed=42,
    num_workers=4,
)

model = Net(num_classes, attr_dim).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

checkpoint_path = Path("artifacts/improved_bird_model.pt")
if checkpoint_path.exists():
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state"])
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {state.get('epoch')}, val_acc={state.get('val_acc'):.4f})")

ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
attr_loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, train_loader, optimizer, scheduler, device, ce_loss, attr_loss_fn, ATTR_WEIGHT, use_amp)
    total_loss = 0
    correct = 0
    
    val_loss, val_acc = evaluate(
        model, val_loader, device, ce_loss, attr_loss_fn, args.attr_weight, use_amp
    )
    
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss:.1f} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")
    
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
