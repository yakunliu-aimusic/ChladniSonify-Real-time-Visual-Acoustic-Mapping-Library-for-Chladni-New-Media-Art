# model/dataset.py
import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ChladniDataset(Dataset):
    def __init__(self, image_dir: str, label_path: str, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform or transforms.ToTensor()
        
        with open(label_path, "r") as f:
            self.labels_dict = json.load(f)
        
        # Filter out missing images for safety
        self.filenames = []
        for fname in self.labels_dict.keys():
            if (self.image_dir / fname).exists():
                self.filenames.append(fname)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path = self.image_dir / fname
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        label = self.labels_dict[fname]["modal_idx"]  # class index: 0~14
        return image, label
    




# dataset.py
# evaluate.py
# network.py
# train.py