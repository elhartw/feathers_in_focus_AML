import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from Dataset import BirdDataset
from models.attribute_cnn import AttributeCNN
import numpy as np


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ---- SETTINGS ----
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20
NUM_CLASSES = 200

# ---- TRANSFORMS ----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- DATA ----
train_dataset = BirdDataset(
    csv_path="data/train_images.csv",
    img_dir="data/",
    transform=train_transform,
    test_mode=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---- MODEL ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttributeCNN(num_classes=NUM_CLASSES, num_attributes=312).to(device)

# Load attributes.npy
attr_matrix = np.load("data/attributes.npy")  # shape (200, 312)
model.load_class_attributes(attr_matrix)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

"""
# ---- MODEL ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

"""

# ---- TRAINING LOOP ----
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        correct += (predicted == labels).sum().item()

    acc = correct / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.2f}, Train Acc: {acc:.4f}")

print("Training klaar, opslaan van model...")
print("Device:", device)
print("Aantal parameters:", sum(p.numel() for p in model.parameters()))

torch.save(model.state_dict(), "attributes_cnn.pth")
print("Model saved.")
