"""
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
