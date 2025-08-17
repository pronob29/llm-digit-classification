#!/usr/bin/env python3
"""
Main Training Script for Digit Classification
Orchestrates the entire training pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.audio_utils import AudioProcessor, create_audio_processor
from src.data_loader import create_data_loaders, get_dataset_stats
from src.model import create_model, count_parameters, get_model_summary
from src.trainer import Trainer

def setup_device() -> str:
    """Setup and return the best available device."""
    # Force CPU for stability
    device = "cpu"
    print("🚀 Using CPU (forced for stability)")
    return device

def load_config(config_path: str = "config.json") -> dict:
    """Load training configuration."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"📋 Loaded configuration from {config_path}")
    else:
        # Default configuration
        config = {
            "data": {
                "data_dir": "data",
                "feature_type": "mfcc",
                "augment_train": True,
                "batch_size": 32,
                "num_workers": 0
            },
            "model": {
                "model_type": "cnn",
                "dropout_rate": 0.3,
                "num_classes": 10
            },
            "training": {
                "num_epochs": 50,
                "learning_rate": 0.001,
                "weight_decay": 1e-4,
                "use_class_weights": True,
                "save_dir": "models"
            },
            "audio": {
                "sample_rate": 8000,
                "n_mfcc": 13,
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "duration": 1.0
            }
        }
        
        # Save default config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"📋 Created default configuration: {config_path}")
    
    return config

def check_dataset(data_dir: str) -> bool:
    """Check if dataset exists and is properly structured."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Dataset directory not found: {data_path}")
        return False
    
    # Check for train and test directories
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print(f"❌ Dataset structure incomplete. Missing train or test directories.")
        return False
    
    # Check for digit directories
    for split_dir in [train_dir, test_dir]:
        for digit in range(10):
            digit_dir = split_dir / str(digit)
            if not digit_dir.exists():
                print(f"❌ Missing digit directory: {digit_dir}")
                return False
            
            # Check for audio files
            audio_files = list(digit_dir.glob("*.wav"))
            if not audio_files:
                print(f"❌ No audio files found in: {digit_dir}")
                return False
    
    print(f"✅ Dataset structure verified: {data_path}")
    return True

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train digit classification model")
    parser.add_argument("--config", type=str, default="config.json", help="Configuration file path")
    parser.add_argument("--data-dir", type=str, help="Override data directory")
    parser.add_argument("--model-type", type=str, choices=["cnn", "lightweight"], help="Override model type")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--feature-type", type=str, choices=["mfcc", "mel"], help="Override feature type")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    
    args = parser.parse_args()
    
    print("🎯 Digit Classification Training Pipeline")
    print("=" * 50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.model_type:
        config["model"]["model_type"] = args.model_type
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    if args.feature_type:
        config["data"]["feature_type"] = args.feature_type
    if args.no_augment:
        config["data"]["augment_train"] = False
    
    # Print configuration
    print("\n📋 Configuration:")
    print(f"   Data Directory: {config['data']['data_dir']}")
    print(f"   Feature Type: {config['data']['feature_type']}")
    print(f"   Model Type: {config['model']['model_type']}")
    print(f"   Batch Size: {config['data']['batch_size']}")
    print(f"   Epochs: {config['training']['num_epochs']}")
    print(f"   Learning Rate: {config['training']['learning_rate']}")
    print(f"   Data Augmentation: {config['data']['augment_train']}")
    
    # Check dataset
    if not check_dataset(config["data"]["data_dir"]):
        print("\n💥 Please run 'python download_dataset.py' first!")
        sys.exit(1)
    
    # Setup device
    device = setup_device()
    
    # Create audio processor
    print("\n🔧 Creating audio processor...")
    audio_processor = create_audio_processor(config["audio"])
    
    # Create data loaders
    print("\n📊 Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["data"]["batch_size"],
        feature_type=config["data"]["feature_type"],
        augment_train=config["data"]["augment_train"],
        num_workers=config["data"]["num_workers"],
        audio_processor=audio_processor
    )
    
    # Get dataset statistics
    stats = get_dataset_stats(config["data"]["data_dir"])
    print(f"\n📈 Dataset Statistics:")
    for split, split_stats in stats.items():
        print(f"   {split.upper()}:")
        for digit, count in split_stats.items():
            print(f"     {digit}: {count} samples")
    
    # Create model
    print(f"\n🧠 Creating {config['model']['model_type']} model...")
    model = create_model(
        model_type=config["model"]["model_type"],
        feature_type=config["data"]["feature_type"],
        num_classes=config["model"]["num_classes"],
        dropout_rate=config["model"]["dropout_rate"]
    )
    
    # Print model summary
    sample_input = torch.randn(1, 1, 13, 32)  # Example MFCC input
    if config["data"]["feature_type"] == "mel":
        sample_input = torch.randn(1, 1, 128, 32)  # Example mel input
    
    print(get_model_summary(model, sample_input.shape))
    print(f"   Parameters: {count_parameters(model):,}")
    
    # Calculate class weights if requested
    class_weights = None
    if config["training"]["use_class_weights"]:
        print("\n⚖️  Calculating class weights...")
        train_dataset = train_loader.dataset
        class_weights = train_dataset.get_class_weights()
        print(f"   Class weights: {class_weights.tolist()}")
    
    # Create trainer
    print("\n🏋️  Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        class_weights=class_weights
    )
    
    # Train model
    print("\n🚀 Starting training...")
    history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        save_dir=config["training"]["save_dir"],
        save_best=True,
        verbose=True
    )
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    results = trainer.evaluate_model()
    
    print(f"\n🏆 Final Results:")
    print(f"   Accuracy: {results['accuracy']:.2f}%")
    print(f"   Best Validation Accuracy: {trainer.best_val_accuracy:.2f}%")
    
    # Print per-class metrics
    print(f"\n📋 Per-Class Performance:")
    report = results['classification_report']
    for i in range(10):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            print(f"   Digit {i}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    # Save results
    results_path = Path(config["training"]["save_dir"]) / "evaluation_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Convert results recursively
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(v) for v in d]
            else:
                return convert_numpy(d)
        
        json.dump(convert_dict(results), f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Plot training history
    print("\n📊 Plotting training history...")
    history_path = Path(config["training"]["save_dir"]) / "training_history.png"
    trainer.plot_training_history(save_path=str(history_path))
    
    # Plot confusion matrix
    print("\n📊 Plotting confusion matrix...")
    cm_path = Path(config["training"]["save_dir"]) / "confusion_matrix.png"
    trainer.plot_confusion_matrix(results, save_path=str(cm_path))
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Models saved to: {config['training']['save_dir']}")
    print(f"🎯 Best model: {config['training']['save_dir']}/best_model.pth")
    print(f"🚀 Ready for inference!")

if __name__ == "__main__":
    main() 