import os
import gdown
import zipfile
from typing import List

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


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
