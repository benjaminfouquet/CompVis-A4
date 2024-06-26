import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TestDataset(Dataset):
    def __init__(self, candidate_pairs_csv, left_dir, right_dir, transform=None):
        self.candidate_pairs = pd.read_csv(candidate_pairs_csv)
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.transform = transform

    def __len__(self):
        return len(self.candidate_pairs)

    def __getitem__(self, idx):
        left_name = os.path.join(
            self.left_dir, self.candidate_pairs.iloc[idx, 0] + ".jpg"
        )
        right_names = [
            os.path.join(self.right_dir, self.candidate_pairs.iloc[idx, i] + ".jpg")
            for i in range(1, len(self.candidate_pairs.iloc[idx, :]))
        ]
        left_image = np.array(Image.open(left_name))
        right_images = np.array(
            [np.array(Image.open(right_name)) for right_name in right_names]
        )
        # if self.transform:
        #     left_image = self.transform(left_image)
        #     right_images = self.transform(right_images)
        return left_image, right_images
