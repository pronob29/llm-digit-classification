#!/usr/bin/env python3
"""
Live Demo for Digit Classification
Real-time microphone input for testing the trained model.
"""

import torch
import torch.nn as nn
import numpy as np
import pyaudio
import wave
import threading
import time
import queue
import argparse
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
import threading

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.audio_utils import AudioProcessor
from src.model import create_model
from typing import Tuple

class AudioRecorder:
    """Real-time audio recorder for live demo."""
    
    def __init__(
        self,
        sample_rate: int = 8000,
        chunk_size: int = 1024,
        channels: int = 1,
        format_type: int = pyaudio.paFloat32
    ):
        """
        Initialize audio recorder.
        
        Args:
            sample_rate: Audio sample rate
            chunk_size: Audio chunk size
            channels: Number of audio channels
            format_type: Audio format type
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.format_type = format_type
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.recording = False
        self.audio_queue = queue.Queue()
        
    def start_recording(self):
        """Start recording audio."""
        def callback(in_data, frame_count, time_info, status):
            if self.recording:
                self.audio_queue.put(in_data)
            return (in_data, pyaudio.paContinue)
        
        self.stream = self.audio.open(
            format=self.format_type,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=callback
        )
        
        self.recording = True
        self.stream.start_stream()
        print("🎤 Microphone recording started...")
    
    def stop_recording(self):
        """Stop recording audio."""
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("🎤 Microphone recording stopped.")
    
    def get_audio_data(self, duration: float = 1.0) -> np.ndarray:
        """
        Get audio data for specified duration.
        
        Args:
            duration: Duration in seconds
            
        Returns:
            Audio data as numpy array
        """
        samples_needed = int(self.sample_rate * duration)
        audio_data = []
        
        while len(audio_data) < samples_needed:
            try:
                data = self.audio_queue.get(timeout=0.1)
                audio_data.extend(np.frombuffer(data, dtype=np.float32))
            except queue.Empty:
                break
        
        # Convert to numpy array and ensure correct length
        audio_array = np.array(audio_data[:samples_needed])
        
        # Pad if necessary
        if len(audio_array) < samples_needed:
            audio_array = np.pad(audio_array, (0, samples_needed - len(audio_array)), 'constant')
        
        return audio_array

class LiveDigitClassifier:
    """Live digit classifier with real-time processing."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.json",
        device: str = "cpu"
    ):
        """
        Initialize live classifier.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            device: Device to run inference on
        """
        self.device = device
        self.model_path = model_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config["audio"]["sample_rate"],
            n_mfcc=self.config["audio"]["n_mfcc"],
            n_fft=self.config["audio"]["n_fft"],
            hop_length=self.config["audio"]["hop_length"],
            n_mels=self.config["audio"]["n_mels"],
            duration=self.config["audio"]["duration"]
        )
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Initialize audio recorder
        self.recorder = AudioRecorder(
            sample_rate=self.config["audio"]["sample_rate"],
            chunk_size=1024,
            channels=1
        )
        
        # Prediction history
        self.prediction_history = []
        self.confidence_history = []
        
    def _load_model(self) -> nn.Module:
        """Load trained model."""
        # Create model architecture
        model = create_model(
            model_type=self.config["model"]["model_type"],
            feature_type=self.config["data"]["feature_type"],
            num_classes=self.config["model"]["num_classes"],
            dropout_rate=self.config["model"]["dropout_rate"]
        )
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"✅ Model loaded from: {self.model_path}")
        return model
    
    def predict_digit(self, audio_data: np.ndarray) -> Tuple[int, float]:
        """
        Predict digit from audio data.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Tuple of (predicted_digit, confidence)
        """
        # Extract features
        if self.config["data"]["feature_type"] == "mfcc":
            features = self.audio_processor.extract_mfcc(audio_data)
        else:
            features = self.audio_processor.extract_mel_spectrogram(audio_data)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
        features_tensor = features_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(features_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_digit = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_digit].item()
        
        return predicted_digit, confidence
    
    def start_live_demo(self):
        """Start live demo with GUI."""
        self.recorder.start_recording()
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Live Digit Classification Demo")
        self.root.geometry("600x400")
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🎤 Live Digit Classification", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Instructions
        instructions = ("Speak a digit (0-9) clearly into your microphone.\n"
                       "The system will classify it in real-time.")
        instruction_label = ttk.Label(main_frame, text=instructions, 
                                     font=('Arial', 10), justify=tk.CENTER)
        instruction_label.grid(row=1, column=0, columnspan=2, pady=(0, 20))
        
        # Prediction display
        self.prediction_var = tk.StringVar(value="Ready to listen...")
        prediction_label = ttk.Label(main_frame, textvariable=self.prediction_var,
                                   font=('Arial', 24, 'bold'), foreground='blue')
        prediction_label.grid(row=2, column=0, columnspan=2, pady=(0, 10))
        
        # Confidence display
        self.confidence_var = tk.StringVar(value="Confidence: --")
        confidence_label = ttk.Label(main_frame, textvariable=self.confidence_var,
                                   font=('Arial', 12))
        confidence_label.grid(row=3, column=0, columnspan=2, pady=(0, 20))
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(0, 20))
        
        self.start_button = ttk.Button(button_frame, text="Start Listening", 
                                      command=self.start_listening)
        self.start_button.grid(row=0, column=0, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Listening", 
                                     command=self.stop_listening, state='disabled')
        self.stop_button.grid(row=0, column=1, padx=(10, 0))
        
        # Status
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var,
                               font=('Arial', 10), foreground='gray')
        status_label.grid(row=5, column=0, columnspan=2)
        
        # History display
        history_frame = ttk.LabelFrame(main_frame, text="Recent Predictions", padding="10")
        history_frame.grid(row=6, column=0, columnspan=2, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.history_text = tk.Text(history_frame, height=6, width=50, font=('Arial', 9))
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for history
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        history_frame.columnconfigure(0, weight=1)
        
        # Start listening thread
        self.listening = False
        self.listening_thread = None
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start GUI
        self.root.mainloop()
    
    def start_listening(self):
        """Start listening for audio input."""
        self.listening = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_var.set("Status: Listening...")
        
        # Start listening thread
        self.listening_thread = threading.Thread(target=self._listening_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()
    
    def stop_listening(self):
        """Stop listening for audio input."""
        self.listening = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_var.set("Status: Stopped")
    
    def _listening_loop(self):
        """Main listening loop."""
        while self.listening:
            try:
                # Get audio data
                audio_data = self.recorder.get_audio_data(duration=1.0)
                
                # Check if audio has sufficient energy
                audio_energy = np.mean(np.abs(audio_data))
                if audio_energy < 0.01:  # Threshold for silence
                    continue
                
                # Make prediction
                predicted_digit, confidence = self.predict_digit(audio_data)
                
                # Update GUI (thread-safe)
                self.root.after(0, self._update_prediction, predicted_digit, confidence)
                
                # Add to history
                self.prediction_history.append(predicted_digit)
                self.confidence_history.append(confidence)
                
                # Keep only recent history
                if len(self.prediction_history) > 10:
                    self.prediction_history.pop(0)
                    self.confidence_history.pop(0)
                
                time.sleep(0.5)  # Small delay to prevent overwhelming
                
            except Exception as e:
                print(f"Error in listening loop: {e}")
                break
    
    def _update_prediction(self, digit: int, confidence: float):
        """Update prediction display (thread-safe)."""
        self.prediction_var.set(f"Predicted: {digit}")
        self.confidence_var.set(f"Confidence: {confidence:.2%}")
        
        # Update history
        timestamp = time.strftime("%H:%M:%S")
        history_entry = f"[{timestamp}] Digit: {digit} (Confidence: {confidence:.2%})\n"
        self.history_text.insert(tk.END, history_entry)
        self.history_text.see(tk.END)
        
        # Color code based on confidence
        if confidence > 0.8:
            self.prediction_var.set(f"Predicted: {digit} ✅")
        elif confidence > 0.6:
            self.prediction_var.set(f"Predicted: {digit} ⚠️")
        else:
            self.prediction_var.set(f"Predicted: {digit} ❓")
    
    def on_closing(self):
        """Handle window closing."""
        self.listening = False
        self.recorder.stop_recording()
        self.root.destroy()

def main():
    """Main function for live demo."""
    parser = argparse.ArgumentParser(description="Live digit classification demo")
    parser.add_argument("--model", type=str, default="models/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        print("💡 Please train a model first using: python train.py")
        sys.exit(1)
    
    # Check if config exists
    if not Path(args.config).exists():
        print(f"❌ Configuration not found: {args.config}")
        print("💡 Please run training first to generate config.json")
        sys.exit(1)
    
    print("🎤 Live Digit Classification Demo")
    print("=" * 40)
    print(f"📁 Model: {args.model}")
    print(f"⚙️  Config: {args.config}")
    print(f"🖥️  Device: {args.device}")
    print("\n🎯 Starting live demo...")
    
    try:
        # Create classifier
        classifier = LiveDigitClassifier(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
        
        # Start demo
        classifier.start_live_demo()
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 