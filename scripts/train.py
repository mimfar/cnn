#!/usr/bin/env python3
"""
Main training script for CNN models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainers.base_trainer import main as train_base_cnn
from training.trainers.improved_trainer import main as train_improved_cnn

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CNN models')
    parser.add_argument('--model', choices=['simple', 'improved'], 
                       default='simple', help='Model to train')
    parser.add_argument('--config', type=str, help='Config file path')
    
    args = parser.parse_args()
    
    if args.model == 'simple':
        train_simple_cnn()
    elif args.model == 'improved':
        train_improved_cnn()
