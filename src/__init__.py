"""
Digit Classification Package
Audio-based digit classification using deep learning.
"""

from .audio_utils import AudioProcessor, create_audio_processor
from .data_loader import DigitAudioDataset, create_data_loaders, get_dataset_stats
from .model import AudioCNN, LightweightAudioNet, create_model, count_parameters
from .trainer import Trainer

__version__ = "1.0.0"
__author__ = "LLM Coding Challenge"

__all__ = [
    "AudioProcessor",
    "create_audio_processor",
    "DigitAudioDataset", 
    "create_data_loaders",
    "get_dataset_stats",
    "AudioCNN",
    "LightweightAudioNet",
    "create_model",
    "count_parameters",
    "Trainer"
] 