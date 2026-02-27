import torch
from torch.utils.data import Dataset, DataLoader
import os
import SimpleITK as sitk
import numpy as np
import cv2


class LiTS(Dataset):
    def __init__(
        self,
        root,
        lowerbound,
        upperbound,
        train=False,
        dev=False,
        transform=None,
        target_transform=None,
        stage = 1,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.stage = stage

        if train:
            root = os.path.join(root, "train")
        elif dev:
            root = os.path.join(root, "dev")
        else:
            root = os.path.join(root, "test")
        image_folder = os.path.join(root, "image")
        target_folder = os.path.join(root, "target")
        for image, target in zip(
                sorted(os.listdir(image_folder)), 
                sorted(os.listdir(target_folder))):

            image_path = os.path.join(image_folder, image)
            target_path = os.path.join(target_folder, target)
            self.images.append(image_path)
            self.targets.append(target_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.images[idx])
        target = sitk.ReadImage(self.targets[idx])

        target = sitk.GetArrayFromImage(target)
        image = sitk.GetArrayFromImage(image)   

        image = np.clip(image, self.lowerbound, self.upperbound)     

        if self.stage == 1:
            target[target > 0] = 1
            processed_image = image  
        else:
            liver_mask = (target > 0)
            processed_image = np.where(liver_mask, image, -1024) 
            target[target == 1] = 0
            target[target == 2] = 1

        target = target.astype(np.float32)
        processed_image = processed_image.astype(np.float32)

        processed_image = torch.from_numpy(processed_image)
        target = torch.from_numpy(target)

        processed_image = processed_image.repeat(3, 1, 1)
        target = (target > 0).float()

        processed_image = processed_image.float()

        return processed_image, target



