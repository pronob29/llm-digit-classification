"""
Data Loader for Digit Classification Dataset
Handles dataset loading, preprocessing, and batching for training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Optional, Dict
import random
from .audio_utils import AudioProcessor

class DigitAudioDataset(Dataset):
    """Dataset class for digit audio classification."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        feature_type: str = "mfcc",
        augment: bool = False,
        audio_processor: Optional[AudioProcessor] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ("train" or "test")
            feature_type: Type of features to extract ("mfcc" or "mel")
            augment: Whether to apply data augmentation
            audio_processor: Audio processor instance
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.feature_type = feature_type
        self.augment = augment
        
        # Initialize audio processor
        if audio_processor is None:
            self.audio_processor = AudioProcessor()
        else:
            self.audio_processor = audio_processor
        
        # Load dataset info
        self.dataset_info = self._load_dataset_info()
        
        # Load file paths and labels
        self.file_paths, self.labels = self._load_file_paths()
        
        print(f"📊 Loaded {split} dataset: {len(self.file_paths)} samples")
    
    def _load_dataset_info(self) -> Dict:
        """Load dataset information."""
        info_path = self.data_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        else:
            # Fallback info
            return {
                "num_classes": 10,
                "classes": list(range(10)),
                "sample_rate": 8000
            }
    
    def _load_file_paths(self) -> Tuple[List[str], List[int]]:
        """Load file paths and corresponding labels."""
        file_paths = []
        labels = []
        
        split_dir = self.data_dir / self.split
        
        for digit in range(10):
            digit_dir = split_dir / str(digit)
            if digit_dir.exists():
                for audio_file in digit_dir.glob("*.wav"):
                    file_paths.append(str(audio_file))
                    labels.append(digit)
        
        return file_paths, labels
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (features, label)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Process audio and extract features
        _, features = self.audio_processor.process_audio_file(
            file_path,
            extract_features=self.feature_type,
            augment=self.augment
        )
        
        # Convert to tensors
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label])[0]
        
        return features_tensor, label_tensor
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        class_counts = np.zeros(10)
        
        for label in self.labels:
            class_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        weights = 1.0 / (class_counts + 1e-8)
        weights = weights / weights.sum() * len(weights)
        
        return torch.FloatTensor(weights)

def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    feature_type: str = "mfcc",
    augment_train: bool = True,
    num_workers: int = 4,
    audio_processor: Optional[AudioProcessor] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        feature_type: Type of features to extract
        augment_train: Whether to augment training data
        num_workers: Number of worker processes
        audio_processor: Audio processor instance
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = DigitAudioDataset(
        data_dir=data_dir,
        split="train",
        feature_type=feature_type,
        augment=augment_train,
        audio_processor=audio_processor
    )
    
    val_dataset = DigitAudioDataset(
        data_dir=data_dir,
        split="test",
        feature_type=feature_type,
        augment=False,
        audio_processor=audio_processor
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def get_dataset_stats(data_dir: str) -> Dict:
    """
    Get dataset statistics.
    
    Args:
        data_dir: Directory containing the dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    
    for split in ["train", "test"]:
        split_dir = Path(data_dir) / split
        split_stats = {}
        
        for digit in range(10):
            digit_dir = split_dir / str(digit)
            if digit_dir.exists():
                count = len(list(digit_dir.glob("*.wav")))
                split_stats[f"digit_{digit}"] = count
        
        stats[split] = split_stats
    
    return stats

class BalancedSampler:
    """Balanced sampler for handling class imbalance."""
    
    def __init__(self, dataset: DigitAudioDataset):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset to sample from
        """
        self.dataset = dataset
        self.indices_per_class = self._get_indices_per_class()
        self.max_samples_per_class = max(len(indices) for indices in self.indices_per_class.values())
    
    def _get_indices_per_class(self) -> Dict[int, List[int]]:
        """Get indices for each class."""
        indices_per_class = {i: [] for i in range(10)}
        
        for idx, label in enumerate(self.dataset.labels):
            indices_per_class[label].append(idx)
        
        return indices_per_class
    
    def __iter__(self):
        """Iterate over balanced samples."""
        indices = []
        
        for class_idx in range(10):
            class_indices = self.indices_per_class[class_idx]
            
            # Repeat indices to match the class with most samples
            if len(class_indices) < self.max_samples_per_class:
                # Repeat with replacement
                repeated_indices = []
                while len(repeated_indices) < self.max_samples_per_class:
                    repeated_indices.extend(class_indices)
                class_indices = repeated_indices[:self.max_samples_per_class]
            else:
                # Sample without replacement
                class_indices = random.sample(class_indices, self.max_samples_per_class)
            
            indices.extend(class_indices)
        
        # Shuffle the combined indices
        random.shuffle(indices)
        
        for idx in indices:
            yield idx
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return self.max_samples_per_class * 10 