"""
Audio Processing Utilities for Digit Classification
Handles feature extraction, preprocessing, and data augmentation.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional, List
import torch
import torchaudio
from pathlib import Path
import random

class AudioProcessor:
    """Audio processing utilities for digit classification."""
    
    def __init__(
        self,
        sample_rate: int = 8000,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        duration: float = 1.0
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            duration: Target duration in seconds
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.duration = duration
        self.target_length = int(sample_rate * duration)
    
    def load_audio(self, filepath: str) -> np.ndarray:
        """
        Load and resample audio file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Audio array at target sample rate
        """
        # Load audio
        audio, sr = librosa.load(filepath, sr=self.sample_rate)
        
        # Pad or truncate to target length
        if len(audio) < self.target_length:
            # Pad with zeros
            audio = np.pad(audio, (0, self.target_length - len(audio)), 'constant')
        else:
            # Truncate to target length
            audio = audio[:self.target_length]
        
        return audio
    
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio array
            
        Returns:
            MFCC features array
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Normalize MFCC
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        # Pad or truncate to fixed size (32 time steps)
        target_length = 32
        if mfcc.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        elif mfcc.shape[1] > target_length:
            # Truncate
            mfcc = mfcc[:, :target_length]
        
        return mfcc
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram features from audio.
        
        Args:
            audio: Audio array
            
        Returns:
            Mel spectrogram array
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)
        
        # Pad or truncate to fixed size (32 time steps)
        target_length = 32
        if mel_spec.shape[1] < target_length:
            # Pad with zeros
            pad_width = target_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        elif mel_spec.shape[1] > target_length:
            # Truncate
            mel_spec = mel_spec[:, :target_length]
        
        return mel_spec
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        Add Gaussian noise to audio.
        
        Args:
            audio: Audio array
            noise_factor: Noise intensity factor
            
        Returns:
            Audio with added noise
        """
        noise = np.random.normal(0, noise_factor, len(audio))
        return audio + noise
    
    def time_shift(self, audio: np.ndarray, shift_factor: float = 0.1) -> np.ndarray:
        """
        Apply time shifting to audio.
        
        Args:
            audio: Audio array
            shift_factor: Maximum shift factor
            
        Returns:
            Time-shifted audio
        """
        shift = int(len(audio) * shift_factor)
        shift = random.randint(-shift, shift)
        
        if shift > 0:
            # Shift right
            audio = np.pad(audio, (shift, 0), 'constant')[:-shift]
        else:
            # Shift left
            audio = np.pad(audio, (0, -shift), 'constant')[shift:]
        
        return audio
    
    def pitch_shift(self, audio: np.ndarray, steps: int = 2) -> np.ndarray:
        """
        Apply pitch shifting to audio.
        
        Args:
            audio: Audio array
            steps: Number of semitones to shift
            
        Returns:
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
    
    def augment_audio(self, audio: np.ndarray, augment_prob: float = 0.5) -> np.ndarray:
        """
        Apply random augmentations to audio.
        
        Args:
            audio: Audio array
            augment_prob: Probability of applying each augmentation
            
        Returns:
            Augmented audio
        """
        augmented = audio.copy()
        
        # Add noise
        if random.random() < augment_prob:
            augmented = self.add_noise(augmented, noise_factor=0.003)
        
        # Time shift
        if random.random() < augment_prob:
            augmented = self.time_shift(augmented, shift_factor=0.05)
        
        # Pitch shift (less aggressive for digits)
        if random.random() < augment_prob * 0.5:
            steps = random.uniform(-1, 1)
            augmented = self.pitch_shift(augmented, steps=steps)
        
        return augmented
    
    def process_audio_file(
        self,
        filepath: str,
        extract_features: str = "mfcc",
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process audio file and extract features.
        
        Args:
            filepath: Path to audio file
            extract_features: Feature extraction method ("mfcc" or "mel")
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (audio, features)
        """
        # Load audio
        audio = self.load_audio(filepath)
        
        # Apply augmentation if requested
        if augment:
            audio = self.augment_audio(audio)
        
        # Extract features
        if extract_features == "mfcc":
            features = self.extract_mfcc(audio)
        elif extract_features == "mel":
            features = self.extract_mel_spectrogram(audio)
        else:
            raise ValueError(f"Unknown feature extraction method: {extract_features}")
        
        return audio, features
    
    def batch_process(
        self,
        filepaths: List[str],
        extract_features: str = "mfcc",
        augment: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Process multiple audio files in batch.
        
        Args:
            filepaths: List of audio file paths
            extract_features: Feature extraction method
            augment: Whether to apply augmentation
            
        Returns:
            Tuple of (audio_list, features_list)
        """
        audios = []
        features = []
        
        for filepath in filepaths:
            audio, feature = self.process_audio_file(filepath, extract_features, augment)
            audios.append(audio)
            features.append(feature)
        
        return audios, features

def create_audio_processor(config: dict) -> AudioProcessor:
    """
    Create audio processor from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AudioProcessor instance
    """
    return AudioProcessor(
        sample_rate=config.get('sample_rate', 8000),
        n_mfcc=config.get('n_mfcc', 13),
        n_fft=config.get('n_fft', 2048),
        hop_length=config.get('hop_length', 512),
        n_mels=config.get('n_mels', 128),
        duration=config.get('duration', 1.0)
    ) 