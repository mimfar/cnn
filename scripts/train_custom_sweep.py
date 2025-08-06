#!/usr/bin/env python3
"""
Custom training script with sweeping over learning rate, batch size, and dropout rate
"""

import sys
import os
import argparse

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainers.base_trainer import main as train_model


def main():
    dropout_rates = [0.25, 0.5]
    learning_rates = 0.001
    batch_sizes = [64]
    num_epochs = 12

    for dropout_rate in dropout_rates:
        for batch_size in batch_sizes:
            name = f"dropout_{dropout_rate}_lr_{learning_rates}_bs_{batch_size}"
            print(f"🚀 Starting training with experiment name: {name or 'auto-generated'}")
            print(f"📊 Training parameters:")
            print(f"   Epochs: {num_epochs}")
            print(f"   Learning Rate: {learning_rates}")
            print(f"   Batch Size: {batch_size}")
        
            # Run training with custom experiment name and parameters
            history, experiment_dir = train_model(
                experiment_name=name,
                num_epochs=num_epochs,
                learning_rate=learning_rates,
                batch_size=batch_size,
                dropout_rate=dropout_rate
        )
    
            print(f"\n✅ Training completed!")
            print(f"📁 Experiment saved to: {experiment_dir}")
            print(f"🎯 Final accuracy: {history['final_val_accuracy']:.2f}%")


if __name__ == "__main__":
    main() 