#!/usr/bin/env python3
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
