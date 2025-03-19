import os
import requests

GITHUB_BASE_URL = "https://raw.githubusercontent.com/Dileep-kumarc/Brain-Tumor-Models/main/"

def download_model_from_github(filename, expected_size_mb):
    """Download a model file from GitHub and check its integrity."""
    if not os.path.exists(filename):
        url = GITHUB_BASE_URL + filename
        print(f"Downloading {filename} from GitHub...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Verify file size
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
            if file_size < expected_size_mb * 0.8:
                print(f"File size mismatch for {filename}. Expected ~{expected_size_mb}MB but got {file_size:.2f}MB.")
                os.remove(filename)
                raise ValueError("Invalid file size")

            print(f"Successfully downloaded {filename} ({file_size:.2f} MB)")

        except Exception as e:
            print(f"Failed to download {filename}: {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)
            raise

# Ensure correct file download
download_model_from_github("best_mri_classifier.pth", 205)  # Adjust size as needed
