import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print(os.path.dirname(os.path.abspath(__file__)))
from models.base_cnn import BaseCNN

def analyze_saturation_issues():
    """
    Analyze the current model for saturation issues and provide recommendations
    """
    print("🔍 ANALYZING CNN MODEL SATURATION ISSUES")
    print("=" * 60)
    
    # Load training history
    try:
        df = pd.read_csv('./training/history_lr_early_stopping_dropout=0.5.csv', index_col=0)
        print(f"✅ Loaded training history with {len(df)} epochs")
    except:
        print("❌ Could not load training history file")
        return
    
    # 1. Analyze Training vs Validation Performance
    print("\n📊 PERFORMANCE ANALYSIS:")
    print("-" * 30)
    
    final_train_acc = df['train_accuracies'].iloc[-1]
    final_val_acc = df['val_accuracies'].iloc[-1]
    final_train_loss = df['train_losses'].iloc[-1]
    final_val_loss = df['val_losses'].iloc[-1]
    
    print(f"Final Training Accuracy: {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    
    # Calculate overfitting metrics
    accuracy_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss
    
    print(f"\n📈 OVERFITTING INDICATORS:")
    print(f"Accuracy Gap (Train - Val): {accuracy_gap:.2f}%")
    print(f"Loss Gap (Val - Train): {loss_gap:.4f}")
    
    if accuracy_gap > 5:
        print("⚠️  SIGNIFICANT OVERFITTING DETECTED!")
    elif accuracy_gap > 2:
        print("⚠️  MODERATE OVERFITTING DETECTED!")
    else:
        print("✅ Minimal overfitting detected")
    
    # 2. Analyze Learning Rate Impact
    print(f"\n🎯 LEARNING RATE ANALYSIS:")
    print("-" * 30)
    
    lr_changes = df['learning_rates'].diff().abs() > 0
    lr_change_epochs = df[lr_changes].index.tolist()
    
    print(f"Learning rate changes at epochs: {lr_change_epochs}")
    
    # Analyze performance after LR changes
    for epoch in lr_change_epochs:
        if epoch < len(df) - 1:
            before_lr = df.loc[epoch-1, 'val_accuracies']
            after_lr = df.loc[epoch+1, 'val_accuracies']
            improvement = after_lr - before_lr
            print(f"  Epoch {epoch}: Val accuracy {before_lr:.2f}% → {after_lr:.2f}% (Δ{improvement:+.2f}%)")
    
    # 3. Identify Saturation Points
    print(f"\n🔄 SATURATION ANALYSIS:")
    print("-" * 30)
    
    # Calculate improvement rates
    val_acc_improvements = df['val_accuracies'].diff()
    train_acc_improvements = df['train_accuracies'].diff()
    
    # Find epochs with minimal improvement
    saturation_threshold = 0.5  # 0.5% improvement threshold
    saturated_epochs = val_acc_improvements[val_acc_improvements < saturation_threshold].index.tolist()
    
    print(f"Epochs with minimal validation improvement (<{saturation_threshold}%): {saturated_epochs}")
    
    # 4. Model Architecture Analysis
    print(f"\n🏗️  MODEL ARCHITECTURE ANALYSIS:")
    print("-" * 30)
    
    model = BaseCNN()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Analyze parameter distribution
    conv_params = sum(p.numel() for name, p in model.named_parameters() if 'conv' in name)
    fc_params = sum(p.numel() for name, p in model.named_parameters() if 'fc' in name)
    
    print(f"Convolutional parameters: {conv_params:,} ({conv_params/total_params*100:.1f}%)")
    print(f"Fully connected parameters: {fc_params:,} ({fc_params/total_params*100:.1f}%)")
    
    if fc_params > conv_params * 2:
        print("⚠️  WARNING: FC layers dominate parameters - potential bottleneck!")
    
    return df

def identify_specific_issues(df):
    """
    Identify specific issues causing saturation
    """
    print(f"\n🔍 SPECIFIC ISSUES IDENTIFIED:")
    print("-" * 30)
    
    issues = []
    
    # Issue 1: Overfitting
    final_acc_gap = df['train_accuracies'].iloc[-1] - df['val_accuracies'].iloc[-1]
    if final_acc_gap > 5:
        issues.append({
            'type': 'Overfitting',
            'severity': 'High' if final_acc_gap > 8 else 'Medium',
            'description': f'Training accuracy ({df["train_accuracies"].iloc[-1]:.1f}%) significantly higher than validation ({df["val_accuracies"].iloc[-1]:.1f}%)',
            'impact': 'Model memorizes training data, poor generalization'
        })
    
    # Issue 2: Learning rate too low
    final_lr = df['learning_rates'].iloc[-1]
    if final_lr < 1e-5:
        issues.append({
            'type': 'Learning Rate Too Low',
            'severity': 'High',
            'description': f'Final learning rate ({final_lr:.2e}) is extremely low',
            'impact': 'Model cannot escape local minima, training stalls'
        })
    
    # Issue 3: Insufficient model capacity
    if df['val_accuracies'].iloc[-1] < 75:
        issues.append({
            'type': 'Insufficient Model Capacity',
            'severity': 'Medium',
            'description': f'Validation accuracy ({df["val_accuracies"].iloc[-1]:.1f}%) below 75%',
            'impact': 'Model may be too simple for the task complexity'
        })
    
    # Issue 4: Poor learning rate schedule
    lr_changes = df['learning_rates'].diff().abs() > 0
    if lr_changes.sum() < 2:
        issues.append({
            'type': 'Poor Learning Rate Schedule',
            'severity': 'Medium',
            'description': f'Only {lr_changes.sum()} learning rate changes during training',
            'impact': 'Model may not find optimal learning trajectory'
        })
    
    # Issue 5: Validation loss not decreasing
    val_loss_trend = df['val_losses'].iloc[-3:].diff().mean()
    if val_loss_trend > 0:
        issues.append({
            'type': 'Validation Loss Increasing',
            'severity': 'High',
            'description': f'Validation loss increasing by {val_loss_trend:.4f} on average',
            'impact': 'Model is overfitting, performance degrading'
        })
    
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['type']} ({issue['severity']})")
        print(f"   {issue['description']}")
        print(f"   Impact: {issue['impact']}")
        print()
    
    return issues

def provide_recommendations(issues):
    """
    Provide specific recommendations to resolve saturation issues
    """
    print(f"💡 RECOMMENDATIONS TO RESOLVE SATURATION:")
    print("=" * 60)
    
    recommendations = []
    
    # Check for overfitting
    overfitting_issues = [i for i in issues if 'Overfitting' in i['type']]
    if overfitting_issues:
        recommendations.extend([
            {
                'priority': 'High',
                'category': 'Regularization',
                'recommendation': 'Increase dropout rate from 0.25 to 0.5',
                'implementation': 'model.dropout = nn.Dropout(0.5)',
                'expected_impact': 'Reduce overfitting, improve generalization'
            },
            {
                'priority': 'High',
                'category': 'Regularization',
                'recommendation': 'Add weight decay to optimizer',
                'implementation': 'optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)',
                'expected_impact': 'Prevent large weights, improve generalization'
            },
            {
                'priority': 'Medium',
                'category': 'Data Augmentation',
                'recommendation': 'Add data augmentation (random horizontal flip, rotation)',
                'implementation': 'transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10)',
                'expected_impact': 'Increase effective dataset size, reduce overfitting'
            }
        ])
    
    # Check for learning rate issues
    lr_issues = [i for i in issues if 'Learning Rate' in i['type']]
    if lr_issues:
        recommendations.extend([
            {
                'priority': 'High',
                'category': 'Learning Rate',
                'recommendation': 'Use CosineAnnealingLR scheduler instead of StepLR',
                'implementation': 'scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)',
                'expected_impact': 'Smoother learning rate decay, better convergence'
            },
            {
                'priority': 'Medium',
                'category': 'Learning Rate',
                'recommendation': 'Implement learning rate warmup',
                'implementation': 'Use warmup_scheduler or custom warmup',
                'expected_impact': 'Stable early training, better convergence'
            }
        ])
    
    # Check for capacity issues
    capacity_issues = [i for i in issues if 'Capacity' in i['type']]
    if capacity_issues:
        recommendations.extend([
            {
                'priority': 'High',
                'category': 'Architecture',
                'recommendation': 'Add more convolutional layers',
                'implementation': 'Add conv3, conv4 layers with batch normalization',
                'expected_impact': 'Increase model capacity, better feature extraction'
            },
            {
                'priority': 'Medium',
                'category': 'Architecture',
                'recommendation': 'Add batch normalization after convolutions',
                'implementation': 'self.bn1 = nn.BatchNorm2d(32), self.bn2 = nn.BatchNorm2d(64)',
                'expected_impact': 'Faster training, better gradient flow'
            },
            {
                'priority': 'Medium',
                'category': 'Architecture',
                'recommendation': 'Increase number of filters in conv layers',
                'implementation': 'Conv1: 32→64, Conv2: 64→128',
                'expected_impact': 'More feature maps, better representation'
            }
        ])
    
    # General recommendations
    recommendations.extend([
        {
            'priority': 'Medium',
            'category': 'Training',
            'recommendation': 'Use gradient clipping',
            'implementation': 'torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)',
            'expected_impact': 'Prevent gradient explosion, stable training'
        },
        {
            'priority': 'Low',
            'category': 'Optimization',
            'recommendation': 'Try different optimizers (AdamW, RAdam)',
            'implementation': 'optimizer = optim.AdamW(model.parameters(), lr=0.001)',
            'expected_impact': 'Better optimization, potentially higher accuracy'
        }
    ])
    
    # Group by priority
    high_priority = [r for r in recommendations if r['priority'] == 'High']
    medium_priority = [r for r in recommendations if r['priority'] == 'Medium']
    low_priority = [r for r in recommendations if r['priority'] == 'Low']
    
    print("🚨 HIGH PRIORITY RECOMMENDATIONS:")
    for i, rec in enumerate(high_priority, 1):
        print(f"{i}. [{rec['category']}] {rec['recommendation']}")
        print(f"   Implementation: {rec['implementation']}")
        print(f"   Expected Impact: {rec['expected_impact']}")
        print()
    
    print("⚡ MEDIUM PRIORITY RECOMMENDATIONS:")
    for i, rec in enumerate(medium_priority, 1):
        print(f"{i}. [{rec['category']}] {rec['recommendation']}")
        print(f"   Implementation: {rec['implementation']}")
        print(f"   Expected Impact: {rec['expected_impact']}")
        print()
    
    print("💡 LOW PRIORITY RECOMMENDATIONS:")
    for i, rec in enumerate(low_priority, 1):
        print(f"{i}. [{rec['category']}] {rec['recommendation']}")
        print(f"   Implementation: {rec['implementation']}")
        print(f"   Expected Impact: {rec['expected_impact']}")
        print()
    
    return recommendations

def create_improved_model():
    """
    Create an improved version of the CNN model addressing the identified issues
    """
    print(f"\n🔧 CREATING IMPROVED CNN MODEL:")
    print("=" * 60)
    
    improved_cnn_code = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedCNN(nn.Module):
    """
    Improved CNN implementation addressing saturation issues
    """
    
    def __init__(self, num_classes=10, input_channels=3):
        super(ImprovedCNN, self).__init__()
        
        # First convolutional block with batch normalization
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global average pooling to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with reduced parameters
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Increased dropout for better regularization
        self.dropout = nn.Dropout(0.5)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
'''
    
    print("Key improvements in the new model:")
    print("✅ Added Batch Normalization for stable training")
    print("✅ Increased model capacity (3 conv layers, more filters)")
    print("✅ Global Average Pooling to reduce FC parameters")
    print("✅ Increased dropout (0.5) for better regularization")
    print("✅ Proper weight initialization")
    print("✅ More sophisticated architecture")
    
    return improved_cnn_code

def create_improved_training_config():
    """
    Create improved training configuration
    """
    print(f"\n⚙️  IMPROVED TRAINING CONFIGURATION:")
    print("=" * 60)
    
    improved_config = '''
# Improved training configuration
def get_improved_transforms():
    """Get improved data transforms with augmentation"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 stats
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    return train_transform, val_transform

def get_improved_optimizer(model):
    """Get improved optimizer with weight decay"""
    return optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

def get_improved_scheduler(optimizer, num_epochs):
    """Get improved learning rate scheduler"""
    return lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def improved_training_step(model, data, targets, optimizer, criterion):
    """Improved training step with gradient clipping"""
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss, outputs
'''
    
    print("Key improvements in training:")
    print("✅ Data augmentation (flip, rotation, color jitter)")
    print("✅ Better normalization (CIFAR-10 specific stats)")
    print("✅ AdamW optimizer with weight decay")
    print("✅ Cosine annealing learning rate scheduler")
    print("✅ Gradient clipping for stability")
    
    return improved_config

if __name__ == "__main__":
    # Run the analysis
    df = analyze_saturation_issues()
    issues = identify_specific_issues(df)
    recommendations = provide_recommendations(issues)
    
    # Create improved model and config
    improved_model = create_improved_model()
    improved_config = create_improved_training_config()
    
    print(f"\n📝 SUMMARY:")
    print("=" * 60)
    print(f"• Identified {len(issues)} specific issues")
    print(f"• Provided {len(recommendations)} recommendations")
    print(f"• Created improved model architecture")
    print(f"• Created improved training configuration")
    print(f"\nNext steps: Implement the high-priority recommendations!") 