# Custom CNN Implementation

A comprehensive implementation of Convolutional Neural Networks (CNN) from scratch using PyTorch, focusing on modern best practices and educational value.

## Project Overview

This project implements a custom CNN architecture with the following features:
- Modular CNN building blocks (Conv2d, BatchNorm, ReLU, etc.)
- Custom loss functions
- Advanced training techniques (learning rate scheduling, regularization)
- Comprehensive evaluation metrics
- Experiment tracking and visualization

## Project Structure

```
cnn_project/
├── data/               # Dataset handling and data loaders
├── models/             # CNN model architectures
├── training/           # Training scripts and utilities
├── utils/              # Helper functions and utilities
├── configs/            # Configuration files
├── experiments/        # Experiment tracking and results
├── notebooks/          # Jupyter notebooks for exploration
└── README.md          # This file
```

## Features

### Model Components
- **Custom Conv2d**: Implementation with configurable parameters
- **Batch Normalization**: For stable training
- **Activation Functions**: ReLU, LeakyReLU, ELU
- **Pooling Layers**: MaxPool2d, AvgPool2d
- **Dropout**: For regularization
- **Attention Mechanisms**: Basic spatial attention

### Training Features
- Learning rate scheduling (StepLR, CosineAnnealingLR)
- Multiple optimizers (SGD, Adam, AdamW)
- Regularization techniques
- Gradient clipping
- Early stopping
- Model checkpointing

### Evaluation
- Multiple metrics (accuracy, precision, recall, F1)
- Confusion matrix visualization
- Training/validation curves
- Model performance analysis

## Getting Started

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install matplotlib seaborn pandas numpy
pip install wandb tensorboard
pip install jupyter
```

### Quick Start
```bash
# Train a basic CNN
python training/train.py --config configs/basic_cnn.yaml

# Train with custom architecture
python training/train.py --config configs/advanced_cnn.yaml

# Evaluate a trained model
python training/evaluate.py --model_path experiments/best_model.pth
```

## Model Architectures

### Basic CNN
- 3 Conv2d layers with BatchNorm and ReLU
- MaxPool2d for downsampling
- Dropout for regularization
- Fully connected classifier

### Advanced CNN
- Deeper architecture with residual connections
- Attention mechanisms
- Advanced regularization
- Custom loss functions

## Configuration

Models and training parameters are configured via YAML files in the `configs/` directory. This allows for easy experimentation and reproducibility.

## Experiment Tracking

The project integrates with Weights & Biases (wandb) for experiment tracking, allowing you to:
- Track training metrics
- Compare different configurations
- Visualize model performance
- Share results with team members

## Contributing

Feel free to extend this project by:
- Adding new model architectures
- Implementing additional training techniques
- Creating new evaluation metrics
- Adding support for different datasets

## License

This project is for educational purposes. Feel free to use and modify as needed. 