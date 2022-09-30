from pathlib import Path

import albumentations as albu
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset as BaseDataset


class Dataset(BaseDataset):
    def __init__(
            self,
            data_path: Path,
            names: list[str],
            transform: callable = None,
            subdir: str = 'train',
    ):
        self.data_path = data_path
        self.names = names
        self.transform = transform
        self.subdir = subdir

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        im = self.data_path / 'img_dir' / self.subdir / self.names[idx]
        img = cv2.imread(str(im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = self.data_path / 'ann_dir' / self.subdir / self.names[idx]
        mask = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)

        if self.transform:
            augment = self.transform(image=img, mask=mask)
            img, mask = augment['image'], augment['mask']

        return img, mask


def get_train_transform(crop: int = 512):
    train_transform = albu.Compose([

        # Pad image if it is small
        albu.PadIfNeeded(min_height=crop, min_width=crop),

        # Geometric transform, mask value is set to -1 because of ignore index in loss
        albu.Flip(),
        albu.ShiftScaleRotate(
            shift_limit=0, scale_limit=0, rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=-1,
        ),

        # Crop the mask with a probability of 0.9, otherwise random crop
        albu.CropNonEmptyMaskIfExists(height=crop, width=crop, p=0.9),
        albu.RandomCrop(height=crop, width=crop),

        # Preprocess transform
        albu.Normalize(),
        ToTensorV2()
    ])
    return train_transform


def get_test_transform():
    test_transform = albu.Compose([
        albu.PadIfNeeded(
            min_height=None,
            min_width=None,
            pad_height_divisor=32,
            pad_width_divisor=32,
            position='top_left'
        ),
        albu.Normalize(),
        ToTensorV2()
    ])
    return test_transform