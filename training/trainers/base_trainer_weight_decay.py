#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json
# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.base_cnn import BaseCNN


def create_experiment_directory(experiment_name=None):
    """
    Create a unique experiment directory with timestamp
    """
    if experiment_name is None:
        # Auto-generate experiment name based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Create experiment directory
    # Use absolute path to ensure experiments are created in the correct location
    experiments_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'experiments')
    experiment_dir = os.path.join(experiments_root, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(f"{experiment_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{experiment_dir}/logs", exist_ok=True)
    os.makedirs(f"{experiment_dir}/plots", exist_ok=True)
    
    print(f"📁 Created experiment directory: {experiment_dir}")
    return experiment_dir


def save_experiment_config(experiment_dir, config):
    """
    Save experiment configuration to JSON file
    """
    config_file = f"{experiment_dir}/config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"📝 Saved experiment config to: {config_file}")


def get_model_info(model):
    """
    Get model architecture information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        'model_name': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'architecture': str(model)
    }
    
    return model_info


def train_model(model, train_loader, val_loader, optimizer, scheduler, experiment_dir, num_epochs=10, patience=2, min_delta=0.005):
    """
    Basic training function with validation and early stopping
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Step the scheduler 
        scheduler.step()
        
        # Store metrics
        current_lr = scheduler.get_last_lr()[0]

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        learning_rates.append(current_lr)
       
        # Early stopping check
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f'  → New best validation loss: {best_val_loss:.4f}')
            
            # Save best model (overwrite previous best)
            best_model_path = f"{experiment_dir}/checkpoints/best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': current_lr
            }, best_model_path)
            print(f'  → Best model saved to {best_model_path}')
        else:
            patience_counter += 1
            print(f'  → No improvement for {patience_counter} epochs')
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, '
              f'Train Acc: {train_accuracy:.2f}%, Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.6f}')
        
        # Check early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs!')
            break
    
    # Save final model
    final_model_path = f"{experiment_dir}/checkpoints/final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_train_loss': avg_train_loss,
        'final_val_loss': avg_val_loss,
        'final_train_accuracy': train_accuracy,
        'final_val_accuracy': val_accuracy,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'learning_rates': learning_rates
        }
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'final_val_accuracy': val_accuracy,
        'experiment_dir': experiment_dir
    }


def load_model(model, device=None, filepath=None):
    if filepath is None:
        # Use absolute path to ensure model is loaded from the correct location
        filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'training', 'checkpoints', 'best_model.pth')
    """
    Load a saved model checkpoint
    """
    if not os.path.exists(filepath):
        print(f"Checkpoint file {filepath} not found!")
        return None
    
    # Load checkpoint to the specified device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {filepath}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.2f}%")
    print(f"Learning rate: {checkpoint['learning_rate']:.6f}")
    
    return checkpoint


def save_final_model(model, history, experiment_dir):
    """
    Save the final model with training history
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = f"{experiment_dir}/checkpoints/final_model_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'final_train_loss': history['train_losses'][-1],
        'final_val_loss': history['val_losses'][-1],
        'final_train_accuracy': history['train_accuracies'][-1],
        'final_val_accuracy': history['val_accuracies'][-1],
        'training_history': history
    }, filepath)
    print(f"Final model saved to {filepath}")


def plot_training_history(history, experiment_dir):
    """
    Create comprehensive training visualization plots
    """
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Decay
    ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    ax3.set_title('Learning Rate Decay', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for better visualization
    
    # Plot 4: Loss vs Accuracy (combined view)
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    line2 = ax4_twin.plot(epochs, history['val_accuracies'], 'r-', label='Val Accuracy', linewidth=2)
    ax4.set_title('Loss vs Accuracy Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('Accuracy (%)', color='r')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='center right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"{experiment_dir}/plots/training_history_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()
    # plt.close()
    
    print(f"Training plots saved to {plot_path}")


#%%
def main(experiment_name=None, num_epochs=12, learning_rate=0.001, batch_size=128, dropout_rate=0.5, weight_decay=0.0001, patience=2):
    """
    Main function to set up data and train the model
    """
    # Create experiment directory
    experiment_dir = create_experiment_directory(experiment_name)
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Clean transforms for validation (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR-10 dataset
    # Use absolute path to ensure data is loaded from the correct location
    data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')
    train_dataset = datasets.CIFAR10(root=data_root, train=True, 
                                    download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_root, train=False, 
                                    download=True, transform=val_transform)

    # Create data loaders with optimization
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=False)

    # Initialize model
    model = BaseCNN(num_classes=10, input_channels=3,dropout_rate=dropout_rate)
    # Ensure model is on the correct device first
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Get model information
    model_info = get_model_info(model)
    print(f"📊 Model Info: {model_info['model_name']} with {model_info['total_parameters']:,} parameters")
    
    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    # Create experiment configuration
    config = {
        'experiment_name': experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'model': model_info,
        'training': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'weight_decay': weight_decay,
            'optimizer': 'Adam',
            'scheduler': 'StepLR',
            'step_size': 4,
            'gamma': 0.1,
            'patience': patience,
            'min_delta': 0.005
        },
        'data': {
            'dataset': 'CIFAR-10',
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'train_transform': 'data_augmentation',
            'val_transform': 'basic_normalization',
            'augmentation': {
                'random_horizontal_flip': True,
                'random_rotation': 10,
                'random_crop': {'size': 32, 'padding': 4},
                'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1}
            }
        },
        'device': str(device)
    }
    
    # Save experiment configuration
    save_experiment_config(experiment_dir, config)
    
    # Train the model
    history = train_model(model, train_loader, val_loader, optimizer, scheduler, experiment_dir, num_epochs=num_epochs, patience=patience)

    print("Training completed!")
    print(f"Final validation accuracy: {history['final_val_accuracy']:.2f}%")
    
    # Save final model and history
    save_final_model(model, history, experiment_dir)
    
    # Save training history to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = f"{experiment_dir}/logs/training_history_{timestamp}.csv"
    df = pd.DataFrame(history)
    df.to_csv(history_file, index=False)
    print(f"📊 Training history saved to: {history_file}")
    
    # Create visualizations
    plot_training_history(history, experiment_dir)
    
    # Save experiment summary
    summary = {
        'experiment_name': config['experiment_name'],
        'timestamp': config['timestamp'],
        'final_val_accuracy': history['final_val_accuracy'],
        'best_val_loss': history['best_val_loss'],
        'total_epochs': len(history['train_losses']),
        'model_parameters': model_info['total_parameters'],
        'experiment_dir': experiment_dir
    }
    
    summary_file = f"{experiment_dir}/experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📋 Experiment summary saved to: {summary_file}")
    
    return history, experiment_dir

if __name__ == "__main__":
    # You can specify an experiment name or let it auto-generate
    main("baseline_weight_decay_test")  # With custom name
    # main()  # Auto-generate name with timestamp
# %%
