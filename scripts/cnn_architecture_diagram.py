import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_cnn_architecture_diagram():
    """
    Create a visual diagram of the BaseCNN architecture
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors for different layer types
    colors = {
        'input': '#E8F4FD',
        'conv': '#FFE6E6',
        'pool': '#E6F3FF',
        'activation': '#F0F8FF',
        'fc': '#FFF2E6',
        'dropout': '#F0FFF0',
        'output': '#FFE6F2'
    }
    
    # Layer definitions with positions and properties
    layers = [
        # Input layer
        {'name': 'Input\n(3×32×32)', 'x': 0.5, 'y': 5, 'width': 1, 'height': 2, 'color': colors['input']},
        
        # First conv block
        {'name': 'Conv1\n(3×3, 32 filters)\n+ Padding=1', 'x': 1.8, 'y': 5, 'width': 1.2, 'height': 2, 'color': colors['conv']},
        {'name': 'ReLU', 'x': 3.2, 'y': 5, 'width': 0.8, 'height': 2, 'color': colors['activation']},
        {'name': 'MaxPool\n(2×2)', 'x': 4.2, 'y': 5, 'width': 1, 'height': 2, 'color': colors['pool']},
        
        # Second conv block
        {'name': 'Conv2\n(3×3, 64 filters)\n+ Padding=1', 'x': 5.5, 'y': 5, 'width': 1.2, 'height': 2, 'color': colors['conv']},
        {'name': 'ReLU', 'x': 6.9, 'y': 5, 'width': 0.8, 'height': 2, 'color': colors['activation']},
        {'name': 'MaxPool\n(2×2)', 'x': 7.9, 'y': 5, 'width': 1, 'height': 2, 'color': colors['pool']},
        
        # Flatten
        {'name': 'Flatten\n(64×8×8 → 4096)', 'x': 9.2, 'y': 5, 'width': 0.8, 'height': 2, 'color': colors['fc']},
    ]
    
    # Fully connected layers (stacked vertically)
    fc_layers = [
        {'name': 'FC1\n(4096 → 128)', 'x': 1.5, 'y': 2.5, 'width': 1.5, 'height': 1.5, 'color': colors['fc']},
        {'name': 'ReLU', 'x': 3.2, 'y': 2.5, 'width': 0.8, 'height': 1.5, 'color': colors['activation']},
        {'name': 'Dropout\n(0.25)', 'x': 4.2, 'y': 2.5, 'width': 1, 'height': 1.5, 'color': colors['dropout']},
        {'name': 'FC2\n(128 → 10)', 'x': 5.5, 'y': 2.5, 'width': 1.5, 'height': 1.5, 'color': colors['fc']},
        {'name': 'Output\n(10 classes)', 'x': 7.2, 'y': 2.5, 'width': 1.2, 'height': 1.5, 'color': colors['output']},
    ]
    
    # Draw all layers
    all_layers = layers + fc_layers
    
    for layer in all_layers:
        # Create rounded rectangle for each layer
        box = FancyBboxPatch(
            (layer['x'], layer['y'] - layer['height']/2),
            layer['width'], layer['height'],
            boxstyle="round,pad=0.1",
            facecolor=layer['color'],
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # Add layer name
        ax.text(layer['x'] + layer['width']/2, layer['y'], 
                layer['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold', wrap=True)
    
    # Add arrows connecting layers
    arrows = [
        # Main flow
        ((1.5, 5), (1.8, 5)),  # Input to Conv1
        ((3.0, 5), (3.2, 5)),  # Conv1 to ReLU
        ((4.0, 5), (4.2, 5)),  # ReLU to MaxPool
        ((5.2, 5), (5.5, 5)),  # MaxPool to Conv2
        ((6.7, 5), (6.9, 5)),  # Conv2 to ReLU
        ((7.7, 5), (7.9, 5)),  # ReLU to MaxPool
        ((8.9, 5), (9.2, 5)),  # MaxPool to Flatten
        
        # FC flow
        ((9.6, 5), (2.25, 3.25)),  # Flatten to FC1
        ((3.0, 2.5), (3.2, 2.5)),  # FC1 to ReLU
        ((4.0, 2.5), (4.2, 2.5)),  # ReLU to Dropout
        ((5.2, 2.5), (5.5, 2.5)),  # Dropout to FC2
        ((7.0, 2.5), (7.2, 2.5)),  # FC2 to Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))
    
    # Add tensor shape annotations
    shapes = [
        {'text': '32×32×3', 'x': 0.5, 'y': 6.5},
        {'text': '32×32×32', 'x': 2.4, 'y': 6.5},
        {'text': '16×16×32', 'x': 4.7, 'y': 6.5},
        {'text': '16×16×64', 'x': 6.1, 'y': 6.5},
        {'text': '8×8×64', 'x': 8.4, 'y': 6.5},
        {'text': '4096', 'x': 9.6, 'y': 6.5},
        {'text': '128', 'x': 2.25, 'y': 4.25},
        {'text': '10', 'x': 7.95, 'y': 4.25},
    ]
    
    for shape in shapes:
        ax.text(shape['x'], shape['y'], shape['text'], 
               ha='center', va='center', fontsize=8, 
               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add section labels
    ax.text(2.5, 7.5, 'Convolutional Layers', ha='center', va='center', 
           fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))
    
    ax.text(2.25, 1.5, 'Fully Connected Layers', ha='center', va='center', 
           fontsize=14, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray'))
    
    # Add title
    ax.text(5, 9, 'BaseCNN Architecture for CIFAR-10 Classification', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Layer'),
        patches.Patch(color=colors['conv'], label='Convolutional Layer'),
        patches.Patch(color=colors['pool'], label='Pooling Layer'),
        patches.Patch(color=colors['activation'], label='Activation Function'),
        patches.Patch(color=colors['fc'], label='Fully Connected Layer'),
        patches.Patch(color=colors['dropout'], label='Dropout'),
        patches.Patch(color=colors['output'], label='Output Layer'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    return fig

def create_detailed_layer_info():
    """
    Create a detailed breakdown of each layer's parameters
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Layer details
    layer_details = [
        {
            'name': 'Input Layer',
            'params': 'CIFAR-10: 32×32×3 RGB images',
            'x': 0.5, 'y': 8.5
        },
        {
            'name': 'Conv1',
            'params': '• Kernel: 3×3\n• Filters: 32\n• Padding: 1\n• Stride: 1\n• Parameters: 896',
            'x': 0.5, 'y': 7
        },
        {
            'name': 'ReLU1',
            'params': '• Activation: ReLU\n• Parameters: 0',
            'x': 0.5, 'y': 5.5
        },
        {
            'name': 'MaxPool1',
            'params': '• Kernel: 2×2\n• Stride: 2\n• Parameters: 0',
            'x': 0.5, 'y': 4
        },
        {
            'name': 'Conv2',
            'params': '• Kernel: 3×3\n• Filters: 64\n• Padding: 1\n• Stride: 1\n• Parameters: 18,496',
            'x': 0.5, 'y': 2.5
        },
        {
            'name': 'ReLU2',
            'params': '• Activation: ReLU\n• Parameters: 0',
            'x': 0.5, 'y': 1
        },
        {
            'name': 'MaxPool2',
            'params': '• Kernel: 2×2\n• Stride: 2\n• Parameters: 0',
            'x': 3, 'y': 8.5
        },
        {
            'name': 'Flatten',
            'params': '• Reshape: 64×8×8 → 4096\n• Parameters: 0',
            'x': 3, 'y': 7
        },
        {
            'name': 'FC1',
            'params': '• Input: 4096\n• Output: 128\n• Parameters: 524,416',
            'x': 3, 'y': 5.5
        },
        {
            'name': 'ReLU3',
            'params': '• Activation: ReLU\n• Parameters: 0',
            'x': 3, 'y': 4
        },
        {
            'name': 'Dropout',
            'params': '• Rate: 0.25\n• Parameters: 0',
            'x': 3, 'y': 2.5
        },
        {
            'name': 'FC2',
            'params': '• Input: 128\n• Output: 10\n• Parameters: 1,290',
            'x': 3, 'y': 1
        },
        {
            'name': 'Total Parameters',
            'params': '545,098 trainable parameters',
            'x': 6, 'y': 8.5
        },
        {
            'name': 'Model Size',
            'params': '~2.1 MB (with optimizer state)',
            'x': 6, 'y': 7
        },
        {
            'name': 'Training Performance',
            'params': '• Validation Accuracy: ~74.18%\n• Training Accuracy: ~81.88%\n• Epochs: 10 (with early stopping)',
            'x': 6, 'y': 5.5
        }
    ]
    
    for detail in layer_details:
        # Create box for each layer detail
        box = FancyBboxPatch(
            (detail['x'] - 0.4, detail['y'] - 0.3),
            0.8, 0.6,
            boxstyle="round,pad=0.05",
            facecolor='lightblue',
            edgecolor='darkblue',
            linewidth=1
        )
        ax.add_patch(box)
        
        # Add layer name
        ax.text(detail['x'], detail['y'] + 0.1, detail['name'], 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add parameters
        ax.text(detail['x'], detail['y'] - 0.1, detail['params'], 
               ha='center', va='center', fontsize=8)
    
    # Add title
    ax.text(5, 9.5, 'BaseCNN Layer Details and Parameters', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create the main architecture diagram
    fig1 = create_cnn_architecture_diagram()
    plt.savefig('cnn_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create the detailed layer information
    fig2 = create_detailed_layer_info()
    plt.savefig('cnn_layer_details.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("CNN architecture diagrams saved as:")
    print("- cnn_architecture_diagram.png")
    print("- cnn_layer_details.png") 