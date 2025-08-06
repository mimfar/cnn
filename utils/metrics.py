"""
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
