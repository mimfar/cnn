#!/usr/bin/env python3
"""
Custom training script with command-line experiment naming
"""

import sys
import os
import argparse

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainers.base_trainer import main as train_model


def main():
    parser = argparse.ArgumentParser(description='Train CNN with custom experiment name')
    parser.add_argument('--name', '-n', type=str, default=None,
                       help='Custom experiment name (default: auto-generate with timestamp)')
    parser.add_argument('--epochs', '-e', type=int, default=12,
                       help='Number of training epochs (default: 12)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', '-b', type=int, default=64,
                       help='Batch size (default: 64)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting training with experiment name: {args.name or 'auto-generated'}")
    print(f"📊 Training parameters:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Batch Size: {args.batch_size}")
    
    # Run training with custom experiment name and parameters
    history, experiment_dir = train_model(
        experiment_name=args.name,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    print(f"\n✅ Training completed!")
    print(f"📁 Experiment saved to: {experiment_dir}")
    print(f"🎯 Final accuracy: {history['final_val_accuracy']:.2f}%")


if __name__ == "__main__":
    main() 