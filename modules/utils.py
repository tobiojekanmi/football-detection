import os
import gdown
import zipfile


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
