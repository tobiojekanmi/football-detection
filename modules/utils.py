import gdown


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
