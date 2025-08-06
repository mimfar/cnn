# 📁 CNN Project File Organization

This document explains the standard file organization for the CNN project.

## 🏗️ Directory Structure

```
cnn/
├── models/                    # Model architectures
│   ├── __init__.py
│   ├── simple_cnn.py         # Simple CNN implementation
│   └── improved_cnn.py       # Improved CNN (to be created)
│
├── data/                     # Dataset storage
│   ├── cifar-10-batches-py/  # CIFAR-10 dataset files
│   └── dataset.py            # Dataset utilities
│
├── training/                 # Training scripts and outputs
│   ├── trainers/             # Training implementations
│   │   ├── __init__.py
│   │   ├── simple_trainer.py # Simple CNN trainer
│   │   └── improved_trainer.py # Improved CNN trainer (to be created)
│   ├── configs/              # Training configurations
│   │   ├── simple_cnn_config.yaml
│   │   └── improved_cnn_config.yaml
│   ├── checkpoints/          # Saved models
│   │   ├── best_model.pth    # Best validation performance
│   │   └── final_model.pth   # Final trained model
│   ├── logs/                 # Training logs and history
│   │   ├── history.csv
│   │   ├── history_lr.csv
│   │   └── history_lr_early_stopping.csv
│   └── plots/                # Training visualizations
│       └── training_history.png
│
├── experiments/              # Experiment tracking and comparison
│   ├── experiment_001/       # Simple CNN baseline
│   │   ├── config.yaml
│   │   ├── results.csv
│   │   └── model.pth
│   ├── experiment_002/       # Improved CNN
│   │   ├── config.yaml
│   │   ├── results.csv
│   │   └── model.pth
│   └── experiment_comparison/
│       └── comparison_report.md
│
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── metrics.py            # Evaluation metrics
│   └── visualization.py      # Plotting utilities
│
├── scripts/                  # Standalone scripts
│   ├── __init__.py
│   ├── train.py             # Main training script
│   ├── evaluate.py          # Model evaluation
│   ├── analyze.py           # Saturation analysis
│   └── cnn_architecture_diagram.py
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   └── test_simple_cnn.py
│
├── notebooks/                # Jupyter notebooks
│   └── model_exploration.py  # Model exploration notebook
│
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── FILE_ORGANIZATION.md     # This file
```

## 📋 File Organization Rules

### 🎯 **Where to Save Different Types of Files:**

#### **Training Outputs:**
- **History CSV files** → `training/logs/`
  - `history.csv`
  - `history_lr.csv`
  - `history_lr_early_stopping.csv`
  - `history_lr_early_stopping_dropout=0.5.csv`

- **Saved Models** → `training/checkpoints/`
  - `best_model.pth` (best validation performance)
  - `final_model.pth` (final trained model)

- **Training Plots** → `training/plots/`
  - `training_history.png`
  - `accuracy_curves.png`
  - `loss_curves.png`

#### **Model Architectures:**
- **CNN Models** → `models/`
  - `simple_cnn.py`
  - `improved_cnn.py`
  - `resnet_cnn.py` (future)

#### **Training Scripts:**
- **Trainer Implementations** → `training/trainers/`
  - `simple_trainer.py`
  - `improved_trainer.py`

- **Main Scripts** → `scripts/`
  - `train.py` (main entry point)
  - `evaluate.py`
  - `analyze.py`

#### **Configuration Files:**
- **Training Configs** → `training/configs/`
  - `simple_cnn_config.yaml`
  - `improved_cnn_config.yaml`

- **Experiment Configs** → `experiments/experiment_XXX/`
  - `config.yaml`

#### **Analysis and Utilities:**
- **Analysis Scripts** → `scripts/`
  - `saturation_analysis.py`
  - `cnn_architecture_diagram.py`

- **Utility Functions** → `utils/`
  - `metrics.py`
  - `visualization.py`

#### **Experiments:**
- **Experiment Results** → `experiments/experiment_XXX/`
  - `config.yaml`
  - `results.csv`
  - `model.pth`
  - `plots/`

## 🔄 **Experiments Directory Purpose:**

The `experiments/` directory is for **experiment tracking and comparison**:

1. **A/B Testing**: Compare different model architectures
2. **Hyperparameter Tuning**: Track different configurations
3. **Reproducibility**: Each experiment has its own config and results
4. **Results Comparison**: Easy to compare performance across experiments

### **Example Experiment Structure:**
```
experiments/
├── experiment_001/           # Simple CNN baseline
│   ├── config.yaml          # Configuration used
│   ├── results.csv          # Training results
│   ├── model.pth            # Trained model
│   └── plots/               # Experiment-specific plots
│       ├── training_curves.png
│       └── confusion_matrix.png
├── experiment_002/           # Improved CNN
│   ├── config.yaml
│   ├── results.csv
│   ├── model.pth
│   └── plots/
└── experiment_comparison/    # Comparison analysis
    ├── comparison_report.md
    └── comparison_plots.png
```

## 🚀 **Usage Examples:**

### **Training a Model:**
```bash
# Train simple CNN
python scripts/train.py --model simple

# Train improved CNN
python scripts/train.py --model improved

# Train with custom config
python scripts/train.py --model simple --config experiments/experiment_001/config.yaml
```

### **Evaluating a Model:**
```bash
# Evaluate best model
python scripts/evaluate.py

# Evaluate specific model
python scripts/evaluate.py --model training/checkpoints/final_model.pth
```

### **Running Analysis:**
```bash
# Analyze saturation issues
python scripts/analyze.py

# Generate architecture diagram
python scripts/cnn_architecture_diagram.py
```

## 📝 **Best Practices:**

1. **Always save training outputs** in the appropriate subdirectories
2. **Use descriptive filenames** for experiments (e.g., `experiment_001_simple_cnn_baseline`)
3. **Keep configurations** with their results for reproducibility
4. **Version control** your experiments but ignore large model files
5. **Document** any changes to the standard structure

## 🔧 **Updating Trainer Paths:**

When creating new trainers, ensure they save files to the correct locations:

```python
# Save history logs
df.to_csv('./training/logs/history.csv')

# Save model checkpoints
torch.save(checkpoint, './training/checkpoints/best_model.pth')

# Save plots
plt.savefig('./training/plots/training_history.png')
```

This organization makes the project scalable, maintainable, and easy to navigate! 