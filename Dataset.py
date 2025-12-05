import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BirdDataset(Dataset):
    def __init__(self, csv_path, img_dir, test_mode=False, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.test_mode = test_mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = os.path.normpath(row['image_path'].lstrip("/"))   

        path = os.path.join(self.img_dir, image_name)
        path = os.path.normpath(path)
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # -- TEST MODE --
        if self.test_mode:
            return image, int(row["id"])

        # -- TRAIN MODE --
        label = int(row["label"]) - 1
        return image, label
