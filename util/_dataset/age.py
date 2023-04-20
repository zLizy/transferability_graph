# https://www.aicrowd.com/showcase/solution-for-submission-175171
import cv2
import numpy as np

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image

class DatasetAge(Dataset):
    """Custom Dataset for loading face images"""

    def __init__(self, csv_path, img_dir, split, transform=None):

        df = pd.read_csv(csv_path)
        self.img_dir = os.path.join(img_dir, split)
        self.image_names = df["ImageID"].values
        self.split = split
        self.csv_path = csv_path
        self.y = [int(int(age.split('-')[0])/10) for age in df['age'].values]
        self.transform = transform

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.img_dir,
                                      self.image_names[index])+".jpg")
        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']
        if self.split != 'test':
            label = self.y[index]
            levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
            levels = torch.tensor(levels, dtype=torch.float32)

            return img, label, levels
        else:
            return img, self.image_names[index]

    def __len__(self):
        return len(self.y)


# train_transforms = A.Compose([
#     A.HorizontalFlip(),
#     A.Rotate(limit=15, p=0.7, interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)),
#     A.Cutout(8, 138, 138, p=0.7),
#     A.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
#     ToTensorV2()
# ])

# train_dataset = DatasetAge(csv_path=TRAIN_CSV_PATH,
#                                img_dir=IMAGE_PATH,
#                                split="train",
#                                transform=train_transforms)

# train_loader = DataLoader(dataset=train_dataset,
#                           batch_size=BATCH_SIZE,
#                           shuffle=True,
#                           num_workers=NUM_WORKERS)