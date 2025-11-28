import os
import gdown
import zipfile
import json
from typing import List, Optional

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from modules.experiment import ExperimentConfig


def download_from_gdrive(link, output_path=None):
    """
    Download file from Google Drive using gdown

    Args:
        link: Google Drive shareable link
        output_path: Optional output file path
    """
    try:
        # Extract file ID from the link
        if "drive.google.com/file/d/" in link:
            file_id = link.split("drive.google.com/file/d/")[1].split("/")[0]
        elif "id=" in link:
            file_id = link.split("id=")[1].split("&")[0]
        else:
            file_id = link

        # Construct the download URL
        url = f"https://drive.google.com/uc?id={file_id}"

        # Download the file
        if output_path:
            gdown.download(url, output_path, fuzzy=True)
        else:
            # Use original filename
            gdown.download(url, fuzzy=True)

        print("File downloaded successfully!")

    except Exception as e:
        print(f"Error downloading file: {e}")


def unzip_file(zip_path: str, extract_to: str):
    """
    Extracts all contents of a ZIP file into the destination folder.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory where files will be extracted.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def predict(
    dataloader,
    model,
    device,
    unnormalize_bbox: bool = True,
) -> pd.DataFrame:
    """
    Run inference on a dataloader and return a DataFrame with:
    image_path, GT bbox, predicted bbox.
    """
    model.eval()
    model.to(device)

    all_records: List[dict] = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Predicting"):
            image_paths, images, gt_bboxes = batch

            images = images.to(device)

            pred_bboxes = model(images)

            gt_bboxes = gt_bboxes.cpu().numpy()
            pred_bboxes = pred_bboxes.cpu().numpy()

            for img_path, gt, pred, img_tensor in zip(
                image_paths, gt_bboxes, pred_bboxes, images
            ):
                # Unnormalize if needed (assuming both gt and pred bboxes were normalized to [0,1])
                if unnormalize_bbox:
                    h, w = img_tensor.shape[1], img_tensor.shape[2]  # C, H, W

                    gt = np.array(
                        [
                            gt[0] * w,
                            gt[1] * h,
                            gt[2] * w,
                            gt[3] * h,
                        ]
                    )
                    pred = np.array(
                        [
                            pred[0] * w,
                            pred[1] * h,
                            pred[2] * w,
                            pred[3] * h,
                        ]
                    )

                rec = {
                    "image_path": img_path,
                    "xmin": int(gt[0]),
                    "ymin": int(gt[1]),
                    "xmax": int(gt[2]),
                    "ymax": int(gt[3]),
                    "pred_xmin": int(pred[0]),
                    "pred_ymin": int(pred[1]),
                    "pred_xmax": int(pred[2]),
                    "pred_ymax": int(pred[3]),
                }

                all_records.append(rec)

    return pd.DataFrame(all_records)


def plot_one(
    image_path: str,
    gt_bbox: np.ndarray,
    pred_bbox: np.ndarray,
    title: Optional[str] = None,
):
    """
    Plots the given image, the groundtruth bounding box and the predicted bounding box.

    Args:
        image_path (str): The path of the image to plot.
        gt_bbox (np.ndarray): The groundtruth bounding box.
        pred_bbox (np.ndarray): The predicted bounding box.
        title (Optional[str]): The title of the plot.
    """

    image = np.array(Image.open(image_path).convert("RGB"))
    plt.figure(figsize=(8, 6))
    plt.imshow(image)

    ax = plt.gca()

    # GT - green
    ax.add_patch(
        plt.Rectangle(  # type: ignore
            (gt_bbox[0], gt_bbox[1]),
            gt_bbox[2] - gt_bbox[0],
            gt_bbox[3] - gt_bbox[1],
            fill=False,
            color="green",
            linewidth=2,
            label="GT",
        )
    )

    # Pred - red
    ax.add_patch(
        plt.Rectangle(  # type: ignore
            (pred_bbox[0], pred_bbox[1]),
            pred_bbox[2] - pred_bbox[0],
            pred_bbox[3] - pred_bbox[1],
            fill=False,
            color="red",
            linewidth=2,
            label="Pred",
        )
    )
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.show()


def load_config(config_path: str) -> ExperimentConfig:
    """
    Helper function to load the experiment config json file
    """

    # Load experiment config
    with open(config_path) as f:
        data = json.load(f)

    return dict_to_dataclass(ExperimentConfig, data)


def dict_to_dataclass(cls, data):
    """
    Helper function to convert nested dictionaries to dataclasses
    """
    if not isinstance(data, dict):
        return data

    # Get the field types from the dataclass
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}

    kwargs = {}
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]

            # Check if the field type is a dataclass and value is a dict
            if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[field_name] = dict_to_dataclass(field_type, value)
            else:
                kwargs[field_name] = value

    return cls(**kwargs)
