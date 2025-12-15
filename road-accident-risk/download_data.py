"""
Download dataset from Kaggle
"""

import os
import sys
import subprocess

def download_kaggle_dataset():
    """Download the Road Accident Risk dataset from Kaggle"""
    
    # Check if kaggle.json exists
    kaggle_json = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json):
        print("Error: Kaggle API credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Move the downloaded kaggle.json to ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Competition name
    competition = 'playground-series-s5e10'
    
    print(f"Downloading dataset from Kaggle competition: {competition}")
    print(f"Target directory: {data_dir}")
    
    try:
        # Download the dataset
        cmd = [
            'kaggle', 'competitions', 'download',
            '-c', competition,
            '-p', data_dir
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Unzip the files
        print("\nUnzipping files...")
        import zipfile
        
        zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip')]
        for zip_file in zip_files:
            zip_path = os.path.join(data_dir, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"Extracted: {zip_file}")
        
        print("\n✓ Dataset downloaded successfully!")
        print(f"\nFiles in {data_dir}:")
        for file in sorted(os.listdir(data_dir)):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                print(f"  - {file} ({size:.2f} MB)")
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print(e.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    download_kaggle_dataset()
