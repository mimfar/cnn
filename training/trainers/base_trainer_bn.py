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

from models.base_cnn_bn import BaseCNNBN
from models.deep_cnn_bn import DeepCNNBN


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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # MixUp function
    def mixup_data(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
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
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate average losses and accuracies
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_accuracy = 100. * train_correct / train_total
        val_accuracy = 100. * val_correct / val_total
        
        # CosineAnnealingLR: step per epoch
        scheduler.step()
        
        # Update learning rate (after last batch step, just read current value)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        learning_rates.append(current_lr)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
        print(f'Learning Rate: {current_lr}')
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Return training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_loss': best_val_loss,
        'final_val_accuracy': val_accuracies[-1]
    }
    
    return history


def save_final_model(model, history, experiment_dir):
    """
    Save the final model state and architecture
    """
    model_file = f"{experiment_dir}/checkpoints/final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'final_val_accuracy': history['final_val_accuracy'],
        'best_val_loss': history['best_val_loss']
    }, model_file)
    print(f"💾 Model saved to: {model_file}")


def plot_training_history(history, experiment_dir):
    """
    Plot training curves
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accuracies'], label='Train Accuracy')
    ax2.plot(history['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_file = f"{experiment_dir}/plots/training_curves.png"
    plt.savefig(plot_file)
    plt.close()
    print(f"📈 Training curves saved to: {plot_file}")


def main(model_type="BaseCNNBN", experiment_name=None, num_epochs=12, learning_rate=0.001, batch_size=128, dropout_rate=0.5, weight_decay=0.0001, patience=4):
    """
    Main function to set up data and train the model
    """
    # Create experiment directory
    experiment_dir = create_experiment_directory(experiment_name)
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
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
    if model_type == "BaseCNNBN":
        model = BaseCNNBN(num_classes=10, input_channels=3, dropout_rate=dropout_rate)
    elif model_type == "DeepCNNBN":
        model = DeepCNNBN(num_classes=10, input_channels=3, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    # Ensure model is on the correct device first
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Training on device: {device}")
    model = model.to(device)
    
    # Get model information
    model_info = get_model_info(model)
    print(f"📊 Model Info: {model_info['model_name']} with {model_info['total_parameters']:,} parameters")
    
    # Initialize optimizer and scheduler
    # SGD with Nesterov momentum tuned for CIFAR-10 style training
    optimizer = optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    # CosineAnnealingLR - smooth cosine decay over epochs
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )
    # Create experiment configuration
    config = {
        'experiment_name': experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'model': model_info,
        'training': {
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': 0.1,
            'dropout_rate': dropout_rate,
            'weight_decay': 5e-4,
            'optimizer': 'SGD',
            'momentum': 0.9,
            'nesterov': True,
            'scheduler': 'CosineAnnealingLR',
            'cosine': {
                'T_max': num_epochs,
                'eta_min': 1e-6
            },
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
                'random_rotation': 10
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
    # You can specify model type and experiment name
    main(model_type="DeepCNNBN", experiment_name="deep_cnn_sgd_cosine")  # Deep model with SGD + CosineAnnealingLR
    # main(model_type="BaseCNNBN", experiment_name="baseline_batchnorm_bs_64")  # Base model
    # main()  # Auto-generate name with timestamp