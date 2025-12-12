"""
Download NYC Taxi Trip Duration dataset from Kaggle.

Usage:
    python download_data.py
"""
import os
import subprocess
import sys
from pathlib import Path

def check_kaggle_installed():
    """Check if kaggle CLI is installed."""
    try:
        subprocess.run(["kaggle", "--version"], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_kaggle_credentials():
    """Check if Kaggle credentials are set up."""
    kaggle_dir = Path.home() / ".kaggle"
    credentials_file = kaggle_dir / "kaggle.json"
    return credentials_file.exists()

def download_dataset():
    """Download the NYC Taxi Trip Duration dataset."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Kaggle competition name
    competition = "nyc-taxi-trip-duration"
    
    print("Checking Kaggle CLI installation...")
    if not check_kaggle_installed():
        print("ERROR: Kaggle CLI is not installed.")
        print("Please install it with: pip install kaggle")
        print("Then download the dataset manually from:")
        print(f"https://www.kaggle.com/c/{competition}/data")
        return False
    
    print("Checking Kaggle credentials...")
    if not check_kaggle_credentials():
        print("WARNING: Kaggle credentials not found.")
        print("Please set up your Kaggle API token:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/")
        print("4. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print(f"Downloading dataset to {data_dir}...")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition, "-p", str(data_dir)],
            check=True
        )
        
        # Extract zip files if they exist
        zip_files = list(data_dir.glob("*.zip"))
        if zip_files:
            print("Extracting zip files...")
            import zipfile
            for zip_file in zip_files:
                print(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(data_dir)
        
        print("Download completed successfully!")
        print(f"Files are in: {data_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to download dataset: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://www.kaggle.com/c/{competition}/data")
        print(f"Extract files to: {data_dir}")
        return False

if __name__ == "__main__":
    success = download_dataset()
    sys.exit(0 if success else 1)
