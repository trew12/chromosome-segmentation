import os
import json

import albumentations as A
from albumentations.pytorch import ToTensorV2
from configs.config import DatasetParams
import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


transform = A.Compose([
    ToTensorV2(),
])

aug_transform = A.Compose([
    A.Flip(p=0.5),
    A.SafeRotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.05), contrast_limit=(-0.1, 0.05), p=0.5),
    ToTensorV2(),
])


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(cfg: DatasetParams):
    annotations_folder = Path(cfg.annotations_folder)
    data_path = list(annotations_folder.rglob('*.json'))
    dataset = ChromosomeDataset(data_path, image_folder=cfg.image_folder, use_aug=cfg.use_aug)
    
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.shuffle, collate_fn=collate_fn)


class ChromosomeDataset(Dataset):
    def __init__(self, annotation_paths, image_folder='dataset/train/images', use_aug=False):
        super().__init__()
        self.annotation_paths = annotation_paths
        self.image_folder = image_folder
        if use_aug:
            self.transform = aug_transform
        else:
            self.transform = transform
        
    def __getitem__(self, index):
        with open(self.annotation_paths[index]) as f:
            data = json.load(f)
        img_path = os.path.join(self.image_folder, data['image_path']) 
        img = cv2.imread(img_path) 
        self.img_size = (data['image_height'], data['image_width'])
        
        masks, labels = self.get_masks(data['shapes'])
        transformed = self.transform(image=img, masks=masks)
        boxes = self.get_boxes(data['shapes'])
        
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32),
                  "labels": torch.as_tensor(labels, dtype=torch.int64),
                  "masks": torch.as_tensor(masks, dtype=torch.uint8)}

        return transformed['image'] / 255, target
                          
    def __len__(self):
        return len(self.annotation_paths)

    def get_boxes(self, shapes):
        boxes = []
        for mask in shapes:
            mask = np.array(mask['points'])
            x_min = round(min(mask[:, 0]))
            x_max = round(max(mask[:, 0]))
            y_min = round(min(mask[:, 1]))
            y_max = round(max(mask[:, 1]))
            boxes.append([x_min , y_min , x_max , y_max])
        boxes = np.stack(boxes, axis=0)
        
        return boxes
    
    def get_masks(self, shapes):
        labels = []
        masks = []
        for mask in shapes:
            labels.append(int(mask['label']))
            binary_mask = np.zeros(self.img_size , dtype=np.uint8)
            polygon_points = np.array(np.array(mask['points']), dtype=np.int64).reshape((-1, 1, 2))
            cv2.fillPoly(binary_mask, [polygon_points], color=1)
            masks.append(binary_mask)
        masks = np.stack(masks, axis=0)
        
        return masks, labels
