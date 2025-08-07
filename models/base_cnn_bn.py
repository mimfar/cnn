import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNNBN(nn.Module):
    """
    A CNN implementation with Batch Normalization.
    Architecture: Conv -> BN -> ReLU -> MaxPool -> Conv -> BN -> ReLU -> MaxPool -> FC -> BN -> ReLU -> FC
    """
    
    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.5):
        super(BaseCNNBN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Max pooling layer (shared)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming 32x32 input (CIFAR-10)
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for fully connected layer
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # First conv block with BatchNorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Second conv block with BatchNorm
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with BatchNorm
        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        """
        features = []
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        features.append(x.clone())
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        features.append(x.clone())
        x = self.pool(x)
        
        return features