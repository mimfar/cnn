#!/usr/bin/env python3
"""
Script to reorganize the CNN project into a standard structure
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Create the standard directory structure"""
    
    directories = [
        # Training organization
        'training/trainers',
        'training/configs', 
        'training/checkpoints',
        'training/logs',
        'training/plots',
        
        # Experiments
        'experiments/experiment_001',  # Simple CNN baseline
        'experiments/experiment_002',  # Improved CNN
        'experiments/experiment_comparison',
        
        # Utils and scripts
        'utils',
        'scripts',
        'tests',
        
        # Notebooks
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def move_files_to_standard_locations():
    """Move existing files to their standard locations"""
    
    moves = [
        # Move training files
        ('base_trainer.py', 'training/trainers/base_trainer.py'),
        ('training_history.png', 'training/plots/training_history.png'),
        ('history_lr_early_stopping.csv', 'training/logs/history_lr_early_stopping.csv'),
        ('history_lr.csv', 'training/logs/history_lr.csv'),
        ('history.csv', 'training/logs/history.csv'),
        
        # Move model files
        ('best_model.pth', 'training/checkpoints/best_model.pth'),
        ('final_model.pth', 'training/checkpoints/final_model.pth'),
        
        # Move analysis scripts
        ('saturation_analysis.py', 'scripts/saturation_analysis.py'),
        ('cnn_architecture_diagram.py', 'scripts/cnn_architecture_diagram.py'),
        ('test_simple_cnn.py', 'tests/test_simple_cnn.py'),
        
        # Move notebook
        ('pg.py', 'notebooks/model_exploration.py'),
    ]
    
    for src, dst in moves:
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✅ Moved {src} → {dst}")
        else:
            print(f"⚠️  Source file not found: {src}")

def create_standard_files():
    """Create standard project files"""
    
    # Create __init__.py files
    init_files = [
        'models/__init__.py',
        'training/trainers/__init__.py',
        'utils/__init__.py',
        'scripts/__init__.py',
        'tests/__init__.py',
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created {init_file}")
    
    # Create main training script
    main_trainer = '''#!/usr/bin/env python3
"""
Main training script for CNN models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainers.base_trainer import main as train_base_cnn
from training.trainers.improved_trainer import main as train_improved_cnn

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN models')
    parser.add_argument('--model', choices=['simple', 'improved'], 
                       default='simple', help='Model to train')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    if args.model == 'simple':
        train_simple_cnn()
    elif args.model == 'improved':
        train_improved_cnn()
'''
    
    with open('scripts/train.py', 'w') as f:
        f.write(main_trainer)
    print("✅ Created scripts/train.py")
    
    # Create evaluation script
    eval_script = '''#!/usr/bin/env python3
"""
Model evaluation script
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_cnn import BaseCNN
from utils.metrics import evaluate_model

def main():
    # Load model
    model = BaseCNN()
    checkpoint = torch.load('training/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    results = evaluate_model(model)
    print(f"Model accuracy: {results['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
'''
    
    with open('scripts/evaluate.py', 'w') as f:
        f.write(eval_script)
    print("✅ Created scripts/evaluate.py")

def create_experiment_configs():
    """Create experiment configuration files"""
    
    # Experiment 001: Simple CNN baseline
    exp001_config = '''# Experiment 001: Simple CNN Baseline
model:
          name: "BaseCNN"
  num_classes: 10
  input_channels: 3

training:
  batch_size: 64
  num_epochs: 10
  learning_rate: 0.001
  optimizer: "Adam"
  scheduler: "StepLR"
  step_size: 4
  gamma: 0.1

data:
  dataset: "CIFAR-10"
  train_transform: "basic"
  val_transform: "basic"

regularization:
  dropout: 0.25
  weight_decay: 0.0
'''
    
    with open('experiments/experiment_001/config.yaml', 'w') as f:
        f.write(exp001_config)
    print("✅ Created experiments/experiment_001/config.yaml")
    
    # Experiment 002: Improved CNN
    exp002_config = '''# Experiment 002: Improved CNN
model:
  name: "ImprovedCNN"
  num_classes: 10
  input_channels: 3

training:
  batch_size: 64
  num_epochs: 15
  learning_rate: 0.001
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  weight_decay: 1e-4

data:
  dataset: "CIFAR-10"
  train_transform: "augmented"
  val_transform: "basic"

regularization:
  dropout: 0.5
  weight_decay: 1e-4
  gradient_clipping: 1.0
'''
    
    with open('experiments/experiment_002/config.yaml', 'w') as f:
        f.write(exp002_config)
    print("✅ Created experiments/experiment_002/config.yaml")

def create_utils_files():
    """Create utility files"""
    
    # Metrics utility
    metrics_utils = '''"""
Utility functions for model evaluation and metrics
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluate model on test dataset
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def calculate_model_size(model):
    """
    Calculate model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb
'''
    
    with open('utils/metrics.py', 'w') as f:
        f.write(metrics_utils)
    print("✅ Created utils/metrics.py")
    
    # Visualization utility
    viz_utils = '''"""
Utility functions for visualization
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_training_history(history, save_path=None):
    """
    Plot training history
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Training and validation loss
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training and validation accuracy
    axes[0, 1].plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(epochs, history['learning_rates'], 'g-')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True)
    
    # Overfitting indicator
    accuracy_gap = [t - v for t, v in zip(history['train_accuracies'], history['val_accuracies'])]
    axes[1, 1].plot(epochs, accuracy_gap, 'purple')
    axes[1, 1].set_title('Overfitting Indicator (Train - Val Accuracy)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Gap (%)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
'''
    
    with open('utils/visualization.py', 'w') as f:
        f.write(viz_utils)
    print("✅ Created utils/visualization.py")

def main():
    """Main reorganization function"""
    print("🔄 REORGANIZING CNN PROJECT STRUCTURE")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Move existing files
    move_files_to_standard_locations()
    
    # Create standard files
    create_standard_files()
    
    # Create experiment configs
    create_experiment_configs()
    
    # Create utility files
    create_utils_files()
    
    print("\n✅ PROJECT REORGANIZATION COMPLETE!")
    print("\n📁 New structure:")
    print("├── models/          # Model architectures")
    print("├── data/            # Dataset storage")
    print("├── training/        # Training scripts and outputs")
    print("│   ├── trainers/    # Training implementations")
    print("│   ├── configs/     # Training configurations")
    print("│   ├── checkpoints/ # Saved models")
    print("│   ├── logs/        # Training logs")
    print("│   └── plots/       # Training visualizations")
    print("├── experiments/     # Experiment tracking")
    print("├── utils/           # Utility functions")
    print("├── scripts/         # Standalone scripts")
    print("├── tests/           # Unit tests")
    print("└── notebooks/       # Jupyter notebooks")
    
    print("\n🎯 Next steps:")
    print("1. Run: python scripts/train.py --model simple")
    print("2. Run: python scripts/evaluate.py")
    print("3. Check experiments/ for A/B testing")

if __name__ == "__main__":
    main() 