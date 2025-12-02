import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BirdDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row['image_path']
        label = int(row['label'])

        path = os.path.join(self.img_dir, image_name)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
