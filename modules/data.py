from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class FootballDatasetConfig:
    """
    Configuration for the football detection dataset.
    """

    # Base directory containing the dataset
    root_data_path: str

    # Dataset split category: "train", "val", or "test"
    dataset_category: str = "train"

    # Target image size (Height, Width). If None, keep original size.
    image_size: Optional[Tuple[int, int]] = None

    # Whether to normalize image pixel values (mean/std if True, to [0, 1] if False)
    normalize_image: bool = True

    # Whether to normalize bounding boxes (i.e., convert to 0–1 range)
    normalize_bbox: bool = True

    # Whether to use additional dataset augmentation
    use_augmentation: bool = False


class FootballDataset(Dataset):
    """
    Football detection dataset.
    """

    def __init__(self, config: FootballDatasetConfig):
        super().__init__()

        self.config = config
        self.dataset_dir = os.path.join(config.root_data_path, config.dataset_category)

        # Validate paths
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # Load and filter annotations
        self.data = self._read_data()

        # Build transforms - always use Albumentations
        self.transforms = self._build_transforms()

    def _read_data(self) -> List[Tuple[str, List[float]]]:
        """
        Load and parse the _annotations CSV file.
        """
        csv_path = os.path.join(self.dataset_dir, "_annotations.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotations file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df = df[df["class"] == "Location-of-ball"]

        data = []
        for _, row in df.iterrows():
            image_path = os.path.join(self.dataset_dir, row["filename"])
            if not os.path.exists(image_path):
                print(f"Warning: Image not found, skipping: {image_path}")
                continue

            # Convert bbox to list and validate
            bbox = [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            if not self._is_valid_bbox(bbox):
                print(f"Warning: Invalid bbox {bbox} for image {image_path}, skipping")
                continue

            data.append((image_path, bbox))

        if not data:
            raise ValueError("No valid data found in the dataset")

        return data

    def _is_valid_bbox(self, bbox: List[float]) -> bool:
        """
        Check if bounding box is valid.
        """
        xmin, ymin, xmax, ymax = bbox
        return (
            xmin < xmax
            and ymin < ymax
            and xmin >= 0
            and ymin >= 0
            and xmax > 0
            and ymax > 0
        )

    def _build_transforms(self) -> A.Compose:
        """
        Build the image transformation pipeline.
        """
        transforms = []

        # Apply augmentations
        if self.config.use_augmentation:
            transforms.extend(
                [
                    # Low risk transforms
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.25),  # Less common but safe to apply
                    A.Affine(
                        scale=(0.9, 1.1),  # Zoom In or Magnify
                        rotate=(-10, 10),  # Rotate
                        p=0.5,
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.1, contrast_limit=0.1, p=0.5
                    ),
                ]
            )

        # Resize image & bbox if desired
        if self.config.image_size is not None:
            transforms.append(
                A.Resize(
                    height=self.config.image_size[0], width=self.config.image_size[1]
                )
            )

        # Apply normalization
        if self.config.normalize_image:
            transforms.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # type: ignore
                    std=[0.229, 0.224, 0.225],  # type: ignore
                )
            )

        # Apply tensor conversion by default
        transforms.append(ToTensorV2())

        # Build the transform with bbox parameters
        bbox_params = A.BboxParams(
            format="pascal_voc",
            label_fields=["category_ids"],
            min_visibility=0.3 if self.config.use_augmentation else 0.0,
        )

        return A.Compose(transforms, bbox_params=bbox_params)

    def __len__(self) -> int:
        """
        Return the number of items in the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Return an item in the dataset given its index (idx)
        """
        image_path, bbox = self.data[idx]
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        transformed = self.transforms(
            image=image,
            bboxes=[bbox],
            category_ids=[0],
        )

        image = transformed["image"]
        bbox = transformed["bboxes"][0]

        if not self.config.normalize_bbox:
            return image_path, image, torch.tensor(bbox, dtype=torch.float32)

        # Normalize bbox to [0,1]
        x_min, y_min, x_max, y_max = bbox
        new_h, new_w = image.shape[1], image.shape[2]
        bbox = torch.tensor(
            [
                x_min / new_w,
                y_min / new_h,
                x_max / new_w,
                y_max / new_h,
            ],
            dtype=torch.float32,
        )

        return image_path, image, bbox


@dataclass
class FootballDataLoaderConfig:
    """
    Configuration for the football detection dataloader.
    """

    # Number of samples per batch
    batch_size: int = 8

    # Whether to shuffle the dataset each epoch
    shuffle: bool = True

    # Number of worker processes for data loading
    num_workers: int = 4

    # Whether to use pinned memory to speed up host-to-GPU transfers
    pin_memory: bool = True


class FootballDataLoader(DataLoader):
    def __init__(self, config: FootballDataLoaderConfig, dataset: FootballDataset):
        super().__init__(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
