"""
Neural Network Model for Digit Classification
Lightweight CNN architecture optimized for audio classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class AudioCNN(nn.Module):
    """
    Lightweight CNN for audio digit classification.
    Optimized for MFCC and mel spectrogram features.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        feature_type: str = "mfcc",
        dropout_rate: float = 0.3
    ):
        """
        Initialize AudioCNN model.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            feature_type: Type of input features ("mfcc" or "mel")
            dropout_rate: Dropout rate for regularization
        """
        super(AudioCNN, self).__init__()
        
        self.feature_type = feature_type
        self.num_classes = num_classes
        
        # Adjust architecture based on feature type
        if feature_type == "mfcc":
            # MFCC features: (n_mfcc, time_steps)
            self.feature_height = 13  # Standard MFCC coefficients
            self.feature_width = 32   # Time steps (will be adjusted)
        else:  # mel
            # Mel spectrogram: (n_mels, time_steps)
            self.feature_height = 128  # Mel bands
            self.feature_width = 32    # Time steps (will be adjusted)
        
        # Feature extraction layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch_size, 1, height, width)
        
        # Convolutional layers with batch normalization and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Adaptive pooling to handle variable input sizes
        x = self.adaptive_pool(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (conv1_features, conv2_features)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        conv1_out = F.relu(self.bn1(self.conv1(x)))
        conv2_out = F.relu(self.bn2(self.conv2(self.pool(conv1_out))))
        
        return conv1_out, conv2_out

class LightweightAudioNet(nn.Module):
    """
    Ultra-lightweight model for real-time inference.
    Optimized for speed and minimal memory usage.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        feature_type: str = "mfcc"
    ):
        """
        Initialize LightweightAudioNet.
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            feature_type: Type of input features
        """
        super(LightweightAudioNet, self).__init__()
        
        self.feature_type = feature_type
        
        # Minimal convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output logits
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x

def create_model(
    model_type: str = "cnn",
    feature_type: str = "mfcc",
    num_classes: int = 10,
    dropout_rate: float = 0.3
) -> nn.Module:
    """
    Create model based on configuration.
    
    Args:
        model_type: Type of model ("cnn" or "lightweight")
        feature_type: Type of input features
        num_classes: Number of output classes
        dropout_rate: Dropout rate
        
    Returns:
        Initialized model
    """
    if model_type == "cnn":
        return AudioCNN(
            input_channels=1,
            num_classes=num_classes,
            feature_type=feature_type,
            dropout_rate=dropout_rate
        )
    elif model_type == "lightweight":
        return LightweightAudioNet(
            input_channels=1,
            num_classes=num_classes,
            feature_type=feature_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int]) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (channels, height, width)
        
    Returns:
        Model summary string
    """
    total_params = count_parameters(model)
    
    summary = f"""
Model Summary:
==============
Model Type: {model.__class__.__name__}
Input Size: {input_size}
Total Parameters: {total_params:,}
Feature Type: {getattr(model, 'feature_type', 'Unknown')}
"""
    
    return summary 