from Dataset import BirdDataset
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = BirdDataset(
    csv_path="data/train_images.csv",
    img_dir="data/",
    transform=transform
)

print("Aantal samples:", len(dataset))

img, label = dataset[0]

print("Eerste afbeelding shape:", img.shape)
print("Label:", label)
