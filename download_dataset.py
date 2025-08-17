#!/usr/bin/env python3
"""
Dataset Download Script for Free Spoken Digit Dataset (FSDD)
Downloads and prepares the dataset for training and evaluation.
"""

import os
import sys
import requests
import zipfile
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import json

def download_fsdd_dataset():
    """Download and prepare the Free Spoken Digit Dataset."""
    
    print("🔄 Downloading Free Spoken Digit Dataset...")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Download from the original FSDD repository
        # This is a simpler approach that avoids Hugging Face audio decoding issues
        fsdd_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
        
        print("📥 Downloading dataset from GitHub...")
        response = requests.get(fsdd_url, stream=True)
        response.raise_for_status()
        
        # Save the zip file
        zip_path = data_dir / "fsdd.zip"
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print("📁 Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove the zip file
        zip_path.unlink()
        
        # Find the extracted directory
        extracted_dir = None
        for item in data_dir.iterdir():
            if item.is_dir() and "free-spoken-digit-dataset" in item.name:
                extracted_dir = item
                break
        
        if not extracted_dir:
            raise Exception("Could not find extracted dataset directory")
        
        # Create organized directory structure
        train_dir = data_dir / "train"
        test_dir = data_dir / "test"
        
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Create digit-specific directories
        for digit in range(10):
            (train_dir / str(digit)).mkdir(exist_ok=True)
            (test_dir / str(digit)).mkdir(exist_ok=True)
        
        # Find all audio files
        recordings_dir = extracted_dir / "recordings"
        if not recordings_dir.exists():
            recordings_dir = extracted_dir / "data"
        
        if not recordings_dir.exists():
            raise Exception("Could not find recordings directory")
        
        audio_files = list(recordings_dir.glob("*.wav"))
        print(f"📊 Found {len(audio_files)} audio files")
        
        # Split into train/test (80/20 split)
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(audio_files)
        
        split_idx = int(0.8 * len(audio_files))
        train_files = audio_files[:split_idx]
        test_files = audio_files[split_idx:]
        
        print(f"📈 Train: {len(train_files)} files, Test: {len(test_files)} files")
        
        # Process training files
        print("📁 Processing training data...")
        for i, audio_file in enumerate(tqdm(train_files, desc="Training")):
            # Extract digit from filename (format: X_Y.wav where X is digit, Y is speaker)
            digit = int(audio_file.stem.split('_')[0])
            
            # Read and save audio
            audio, sample_rate = sf.read(audio_file)
            filename = f"train_{i:04d}.wav"
            filepath = train_dir / str(digit) / filename
            sf.write(filepath, audio, sample_rate)
        
        # Process test files
        print("📁 Processing test data...")
        for i, audio_file in enumerate(tqdm(test_files, desc="Test")):
            # Extract digit from filename
            digit = int(audio_file.stem.split('_')[0])
            
            # Read and save audio
            audio, sample_rate = sf.read(audio_file)
            filename = f"test_{i:04d}.wav"
            filepath = test_dir / str(digit) / filename
            sf.write(filepath, audio, sample_rate)
        
        # Clean up extracted directory
        import shutil
        shutil.rmtree(extracted_dir)
        
        # Create dataset info file
        info = {
            "train_samples": len(train_files),
            "test_samples": len(test_files),
            "sample_rate": sample_rate,
            "num_classes": 10,
            "classes": list(range(10))
        }
        
        with open(data_dir / "dataset_info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        print("✅ Dataset preparation completed!")
        print(f"📂 Data saved to: {data_dir.absolute()}")
        
        # Print statistics
        print("\n📈 Dataset Statistics:")
        for split in ['train', 'test']:
            print(f"\n{split.upper()} SET:")
            split_dir = data_dir / split
            for digit in range(10):
                digit_dir = split_dir / str(digit)
                count = len(list(digit_dir.glob("*.wav")))
                print(f"   Digit {digit}: {count} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_fsdd_dataset()
    if success:
        print("\n🎉 Dataset download completed successfully!")
        print("🚀 You can now run: python train.py")
    else:
        print("\n💥 Dataset download failed!")
        sys.exit(1) 