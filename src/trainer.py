"""
Training Module for Digit Classification
Handles training loop, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class Trainer:
    """Training manager for digit classification models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            class_weights: Class weights for imbalanced datasets
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        
        # Early stopping
        self.patience = 10
        self.counter = 0
        
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Progress update
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100. * correct / total:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, List[int], List[int]]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy, predictions, targets)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "models",
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save models
            save_best: Whether to save best model
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Training on device: {self.device}")
        print(f"📁 Model will be saved to: {save_path.absolute()}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, predictions, targets = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Check for best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.counter = 0
                
                if save_best:
                    self.save_model(save_path / "best_model.pth")
            else:
                self.counter += 1
            
            epoch_time = time.time() - epoch_start
            
            if verbose:
                print(f"\n📈 Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
                print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                print(f"   Best Val Acc: {self.best_val_accuracy:.2f}% (Epoch {self.best_epoch+1})")
                print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if self.counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time:.2f} seconds")
        print(f"🏆 Best validation accuracy: {self.best_val_accuracy:.2f}% (Epoch {self.best_epoch+1})")
        
        # Save final model
        self.save_model(save_path / "final_model.pth")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'total_time': total_time
        }
        
        with open(save_path / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_model(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.best_epoch = checkpoint['best_epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.val_accuracies = checkpoint['val_accuracies']
        print(f"📂 Model loaded from: {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Acc', color='blue')
        ax2.plot(self.val_accuracies, label='Val Acc', color='red')
        ax2.axhline(y=self.best_val_accuracy, color='green', linestyle='--', 
                   label=f'Best Val Acc: {self.best_val_accuracy:.2f}%')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        lr_history = [group['lr'] for group in self.optimizer.param_groups]
        ax3.plot(lr_history, color='purple')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(self.train_losses, self.val_losses)]
        ax4.plot(loss_diff, color='orange')
        ax4.set_title('Train-Val Loss Difference')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('|Train Loss - Val Loss|')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training history plot saved to: {save_path}")
        
        plt.show()
    
    def evaluate_model(self) -> Dict:
        """Evaluate model performance."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * sum(p == t for p, t in zip(all_predictions, all_targets)) / len(all_targets)
        
        # Classification report
        report = classification_report(
            all_targets, all_predictions,
            target_names=[str(i) for i in range(10)],
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
        
        return results
    
    def plot_confusion_matrix(self, results: Dict, save_path: Optional[str] = None):
        """Plot confusion matrix."""
        cm = np.array(results['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10)
        )
        plt.title(f'Confusion Matrix (Accuracy: {results["accuracy"]:.2f}%)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.show() 