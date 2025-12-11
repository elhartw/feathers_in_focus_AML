import torch
import torch.nn as nn
import torch.nn.functional as F


class AttributeCNN(nn.Module):
    def __init__(self, num_classes=200, num_attributes=312):
        super().__init__()

        # --- CNN IMAGE ENCODER ---
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Final CNN embedding size
        self.feat_dim = 128 * 28 * 28

        self.fc_img = nn.Linear(self.feat_dim, 512)

        # --- ATTRIBUTE EMBEDDING ---
        # attributes.npy shape: (200 classes, 312 attributes)
        # We convert attributes → 512-dim embeddings
        self.fc_attr = nn.Linear(num_attributes, 512)

        # Store class embeddings (non-trainable)
        self.class_attr_embed = None  # filled after loading attributes.npy

    def load_class_attributes(self, attr_matrix):
        """ attr_matrix = numpy array (200, 312) """
        tensor_attr = torch.tensor(attr_matrix).float()
        with torch.no_grad():
            self.class_attr_embed = self.fc_attr(tensor_attr)   # shape (200, 512)
            self.class_attr_embed = F.normalize(self.class_attr_embed, dim=1)

    def forward(self, x):
        # --- Image encoding ---
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        img_feat = F.relu(self.fc_img(x))    # (B, 512)
        img_feat = F.normalize(img_feat, dim=1)

        # --- Compute similarity with attribute embeddings ---
        # class_attr_embed shape: (200, 512)
        # img_feat shape: (B, 512)
        logits = img_feat @ self.class_attr_embed.T   # (B, 200)

        return logits
