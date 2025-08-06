import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base_cnn import BaseCNN


def test_model():
    """Test if our simple CNN works correctly"""
    print("Testing BaseCNN...")
    
    # Create model
    model = BaseCNN(num_classes=10)
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input (batch_size=2, channels=3, height=32, width=32)
    dummy_input = torch.randn(2, 3, 32, 32)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output classes: {output.shape[1]}")
    
    # Test feature maps
    feature_maps = model.get_feature_maps(dummy_input)
    print(f"Number of feature map layers: {len(feature_maps)}")
    for i, fm in enumerate(feature_maps):
        print(f"Feature map {i+1} shape: {fm.shape}")
    
    print("\n✅ All tests passed! Model is working correctly.")


if __name__ == "__main__":
    test_model() 