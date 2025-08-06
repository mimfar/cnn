import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseCNN(nn.Module):
    """
    A basic CNN implementation for educational purposes.
    Basic architecture: Conv -> ReLU -> MaxPool -> Conv -> ReLU -> MaxPool -> FC
    """
    
    def __init__(self, num_classes=10, input_channels=3,dropout_rate=0.5):
        super(BaseCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Assuming 32x32 input (CIFAR-10)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization
        """
        features = []
        
        # First conv block
        x = F.relu(self.conv1(x))
        features.append(x.clone())
        x = self.pool(x)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        features.append(x.clone())
        x = self.pool(x)
        
        return features 