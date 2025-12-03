import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from Dataset import BirdDataset
from models.simple_cnn import SimpleCNN

# --- Settings ---
NUM_CLASSES = 200
MODEL_PATH = "simple_cnn.pth"

# --- Transform ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# --- Data ---
test_dataset = BirdDataset(
    csv_path="data/test_images_path.csv",
    img_dir="data/",
    transform=test_transform,
    test_mode=True        
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# --- Predict ---
predictions = []
with torch.no_grad():
    for images, img_ids in test_loader:
        images = images.to(device)
        outputs = model(images)
   
        preds = outputs.argmax(dim=1).cpu().numpy() + 1

        predictions.extend([[int(i), int(p)] for i, p in zip(img_ids, preds)])


df = pd.DataFrame(predictions, columns=["id", "label"])
df.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")
