#!/usr/bin/env python3
"""
Experiment Manager - Utility script to manage and compare experiments
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import sys

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def calculate_saturation_metrics(exp_path):
    """
    Calculate saturation metrics from training history
    """
    import glob
    
    # Find the training history file
    history_files = glob.glob(os.path.join(exp_path, "logs", "training_history_*.csv"))
    
    if not history_files:
        return {
            'accuracy_gap': 0,
            'final_train_accuracy': 0,
            'final_val_accuracy': 0,
            'overfitting_score': 0
        }
    
    # Use the most recent history file
    history_file = max(history_files, key=os.path.getctime)
    
    try:
        df = pd.read_csv(history_file)
        
        if len(df) == 0:
            return {
                'accuracy_gap': 0,
                'final_train_accuracy': 0,
                'final_val_accuracy': 0,
                'overfitting_score': 0
            }
        
        # Get final accuracies
        final_train_acc = df['train_accuracies'].iloc[-1] if 'train_accuracies' in df.columns else 0
        final_val_acc = df['val_accuracies'].iloc[-1] if 'val_accuracies' in df.columns else 0
        
        # Calculate accuracy gap (overfitting indicator)
        accuracy_gap = final_train_acc - final_val_acc
        
        # Calculate overfitting score (normalized gap)
        overfitting_score = accuracy_gap / max(final_train_acc, 1) * 100
        
        return {
            'accuracy_gap': accuracy_gap,
            'final_train_accuracy': final_train_acc,
            'final_val_accuracy': final_val_acc,
            'overfitting_score': overfitting_score
        }
        
    except Exception as e:
        print(f"⚠️  Error reading history file {history_file}: {e}")
        return {
            'accuracy_gap': 0,
            'final_train_accuracy': 0,
            'final_val_accuracy': 0,
            'overfitting_score': 0
        }


def list_experiments():
    """
    List all experiments with their details
    """
    experiments_dir = "./experiments"
    if not os.path.exists(experiments_dir):
        print("❌ No experiments directory found!")
        return
    
    experiments = []
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        if os.path.isdir(exp_path):
            # Try to load experiment summary
            summary_file = os.path.join(exp_path, "experiment_summary.json")
            config_file = os.path.join(exp_path, "config.json")
            
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                experiments.append({
                    'name': exp_dir,
                    'summary': summary,
                    'has_config': os.path.exists(config_file)
                })
            else:
                experiments.append({
                    'name': exp_dir,
                    'summary': None,
                    'has_config': os.path.exists(config_file)
                })
    
    if not experiments:
        print("❌ No experiments found!")
        return
    
    print("🔬 EXPERIMENTS FOUND:")
    print("=" * 80)
    
    for exp in experiments:
        print(f"\n📁 {exp['name']}")
        if exp['summary']:
            summary = exp['summary']
            print(f"   📅 Timestamp: {summary.get('timestamp', 'N/A')}")
            print(f"   🎯 Final Accuracy: {summary.get('final_val_accuracy', 'N/A'):.2f}%")
            print(f"   📉 Best Loss: {summary.get('best_val_loss', 'N/A'):.4f}")
            print(f"   🔄 Epochs: {summary.get('total_epochs', 'N/A')}")
            print(f"   🧠 Parameters: {summary.get('model_parameters', 'N/A'):,}")
        else:
            print("   ⚠️  No summary found")
        
        if exp['has_config']:
            print("   ✅ Has configuration")
        else:
            print("   ❌ No configuration")
    
    return experiments


def compare_experiments(exp_names=None):
    """
    Compare multiple experiments with saturation detection
    """
    experiments_dir = "./experiments"
    if not os.path.exists(experiments_dir):
        print("❌ No experiments directory found!")
        return
    
    # Get all experiments if none specified
    if exp_names is None:
        exp_dirs = [d for d in os.listdir(experiments_dir) 
                   if os.path.isdir(os.path.join(experiments_dir, d))]
    else:
        exp_dirs = exp_names
    
    comparisons = []
    
    for exp_dir in exp_dirs:
        exp_path = os.path.join(experiments_dir, exp_dir)
        summary_file = os.path.join(exp_path, "experiment_summary.json")
        config_file = os.path.join(exp_path, "config.json")
        history_file = os.path.join(exp_path, "logs", "training_history_*.csv")
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            config = None
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            
            # Calculate saturation metrics
            saturation_metrics = calculate_saturation_metrics(exp_path)
            
            comparisons.append({
                'name': exp_dir,
                'summary': summary,
                'config': config,
                'saturation': saturation_metrics
            })
    
    if not comparisons:
        print("❌ No valid experiments found for comparison!")
        return
    
    # Create comparison table
    print("📊 EXPERIMENT COMPARISON:")
    print("=" * 120)
    
    # Header
    headers = ["Experiment", "Val Acc", "Train Acc", "Gap", "Saturation", "Best Loss", "Epochs", "Model"]
    print(f"{headers[0]:<20} {headers[1]:<8} {headers[2]:<9} {headers[3]:<6} {headers[4]:<12} {headers[5]:<12} {headers[6]:<8} {headers[7]}")
    print("-" * 120)
    
    for comp in comparisons:
        summary = comp['summary']
        config = comp['config']
        saturation = comp['saturation']
        
        model_name = "Unknown"
        if config and 'model' in config:
            model_name = config['model'].get('model_name', 'Unknown')
            # Update old SimpleCNN references to BaseCNN
            if model_name == "SimpleCNN":
                model_name = "BaseCNN"
        
        # Determine saturation status
        gap = saturation.get('accuracy_gap', 0)
        if gap > 10:
            saturation_status = "🔴 HIGH"
        elif gap > 5:
            saturation_status = "🟡 MEDIUM"
        else:
            saturation_status = "🟢 LOW"
        
        print(f"{comp['name']:<20} "
              f"{summary.get('final_val_accuracy', 0):<8.2f} "
              f"{saturation.get('final_train_accuracy', 0):<9.2f} "
              f"{gap:<6.1f} "
              f"{saturation_status:<12} "
              f"{summary.get('best_val_loss', 0):<12.4f} "
              f"{summary.get('total_epochs', 0):<8} "
              f"{model_name}")
    
    # Create comparison plot
    create_comparison_plot(comparisons)


def create_comparison_plot(comparisons):
    """
    Create a comparison plot of experiments with saturation detection
    """
    if len(comparisons) < 2:
        print("⚠️  Need at least 2 experiments for comparison plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    names = [comp['name'] for comp in comparisons]
    val_accuracies = [comp['summary'].get('final_val_accuracy', 0) for comp in comparisons]
    train_accuracies = [comp['saturation'].get('final_train_accuracy', 0) for comp in comparisons]
    accuracy_gaps = [comp['saturation'].get('accuracy_gap', 0) for comp in comparisons]
    losses = [comp['summary'].get('best_val_loss', 0) for comp in comparisons]
    
    # Create color coding for saturation
    colors = []
    for gap in accuracy_gaps:
        if gap > 10:
            colors.append('red')  # High overfitting
        elif gap > 5:
            colors.append('orange')  # Medium overfitting
        else:
            colors.append('green')  # Low overfitting
    
    # Accuracy comparison (Train vs Val)
    x = range(len(names))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], train_accuracies, width, label='Train Acc', color='lightblue', alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], val_accuracies, width, label='Val Acc', color='darkblue', alpha=0.8)
    
    ax1.set_title('Training vs Validation Accuracy', fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, train_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    for bar, acc in zip(bars2, val_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Accuracy gap (saturation indicator)
    bars3 = ax2.bar(names, accuracy_gaps, color=colors, alpha=0.7)
    ax2.set_title('Accuracy Gap (Overfitting Indicator)', fontweight='bold')
    ax2.set_ylabel('Gap (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Medium Threshold')
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='High Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, gap in zip(bars3, accuracy_gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{gap:.1f}%', ha='center', va='bottom')
    
    # Loss comparison
    bars4 = ax3.bar(names, losses, color='lightcoral', alpha=0.7)
    ax3.set_title('Best Validation Loss', fontweight='bold')
    ax3.set_ylabel('Loss')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, loss in zip(bars4, losses):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{loss:.3f}', ha='center', va='bottom')
    
    # Saturation summary
    saturation_levels = []
    for gap in accuracy_gaps:
        if gap > 10:
            saturation_levels.append('High')
        elif gap > 5:
            saturation_levels.append('Medium')
        else:
            saturation_levels.append('Low')
    
    # Create a text summary
    ax4.axis('off')
    summary_text = "SATURATION ANALYSIS:\n\n"
    for i, (name, gap, level) in enumerate(zip(names, accuracy_gaps, saturation_levels)):
        summary_text += f"{name}:\n"
        summary_text += f"  Gap: {gap:.1f}%\n"
        summary_text += f"  Level: {level}\n"
        if gap > 10:
            summary_text += f"  WARNING: High overfitting detected!\n"
        elif gap > 5:
            summary_text += f"  WARNING: Moderate overfitting\n"
        else:
            summary_text += f"  OK: Good generalization\n"
        summary_text += "\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save comparison plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"./experiments/experiment_comparison/comparison_{timestamp}.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison plot saved to: {plot_path}")


def load_experiment(exp_name):
    """
    Load and display experiment details with saturation analysis
    """
    exp_path = f"./experiments/{exp_name}"
    if not os.path.exists(exp_path):
        print(f"❌ Experiment '{exp_name}' not found!")
        return
    
    print(f"🔍 EXPERIMENT DETAILS: {exp_name}")
    print("=" * 60)
    
    # Load summary
    summary_file = os.path.join(exp_path, "experiment_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        print(f"📅 Timestamp: {summary.get('timestamp', 'N/A')}")
        print(f"🎯 Final Accuracy: {summary.get('final_val_accuracy', 'N/A'):.2f}%")
        print(f"📉 Best Loss: {summary.get('best_val_loss', 'N/A'):.4f}")
        print(f"🔄 Total Epochs: {summary.get('total_epochs', 'N/A')}")
        print(f"🧠 Model Parameters: {summary.get('model_parameters', 'N/A'):,}")
        print(f"📁 Experiment Directory: {summary.get('experiment_dir', 'N/A')}")
    
    # Calculate and display saturation metrics
    saturation = calculate_saturation_metrics(exp_path)
    if saturation['final_train_accuracy'] > 0:
        print(f"\n🔍 SATURATION ANALYSIS:")
        print(f"   📊 Training Accuracy: {saturation['final_train_accuracy']:.2f}%")
        print(f"   📊 Validation Accuracy: {saturation['final_val_accuracy']:.2f}%")
        print(f"   📈 Accuracy Gap: {saturation['accuracy_gap']:.2f}%")
        print(f"   📊 Overfitting Score: {saturation['overfitting_score']:.1f}%")
        
        # Determine saturation level
        gap = saturation['accuracy_gap']
        if gap > 10:
            print(f"   ⚠️  SATURATION LEVEL: 🔴 HIGH - Significant overfitting detected!")
            print(f"      💡 Recommendations: Increase regularization, reduce model complexity, or use early stopping")
        elif gap > 5:
            print(f"   ⚠️  SATURATION LEVEL: 🟡 MEDIUM - Moderate overfitting detected")
            print(f"      💡 Recommendations: Consider adding dropout or reducing learning rate")
        else:
            print(f"   ✅ SATURATION LEVEL: 🟢 LOW - Good generalization")
            print(f"      💡 Model is learning well without significant overfitting")
    
    # Load config
    config_file = os.path.join(exp_path, "config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n⚙️  CONFIGURATION:")
        print(f"   Model: {config.get('model', {}).get('model_name', 'N/A')}")
        print(f"   Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
        print(f"   Batch Size: {config.get('training', {}).get('batch_size', 'N/A')}")
        print(f"   Optimizer: {config.get('training', {}).get('optimizer', 'N/A')}")
        print(f"   Scheduler: {config.get('training', {}).get('scheduler', 'N/A')}")
    
    # List files
    print(f"\n📁 EXPERIMENT FILES:")
    for root, dirs, files in os.walk(exp_path):
        level = root.replace(exp_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")


def delete_specific_experiments(exp_names):
    """
    Delete specific experiments by name
    """
    experiments_dir = "./experiments"
    if not os.path.exists(experiments_dir):
        print("❌ No experiments directory found!")
        return
    
    experiments_to_delete = []
    
    for exp_name in exp_names:
        exp_path = os.path.join(experiments_dir, exp_name)
        if os.path.exists(exp_path):
            experiments_to_delete.append(exp_name)
        else:
            print(f"⚠️  Experiment '{exp_name}' not found!")
    
    if not experiments_to_delete:
        print("❌ No valid experiments to delete!")
        return
    
    print(f"🗑️  EXPERIMENTS TO DELETE:")
    for exp_name in experiments_to_delete:
        exp_path = os.path.join(experiments_dir, exp_name)
        print(f"   {exp_name} ({os.path.getsize(exp_path) / (1024*1024):.1f} MB)")
    
    response = input(f"\n❓ Delete {len(experiments_to_delete)} experiment(s)? (y/N): ")
    if response.lower() == 'y':
        for exp_name in experiments_to_delete:
            exp_path = os.path.join(experiments_dir, exp_name)
            import shutil
            shutil.rmtree(exp_path)
            print(f"🗑️  Deleted: {exp_name}")
        print("✅ Deletion completed!")
    else:
        print("❌ Deletion cancelled.")


def clean_old_experiments(days_old=30):
    """
    Clean experiments older than specified days
    """
    import time
    
    experiments_dir = "./experiments"
    if not os.path.exists(experiments_dir):
        print("❌ No experiments directory found!")
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_old * 24 * 60 * 60)
    
    old_experiments = []
    
    for exp_dir in os.listdir(experiments_dir):
        exp_path = os.path.join(experiments_dir, exp_dir)
        if os.path.isdir(exp_path):
            # Check creation time
            creation_time = os.path.getctime(exp_path)
            if creation_time < cutoff_time:
                old_experiments.append((exp_dir, creation_time))
    
    if not old_experiments:
        print(f"✅ No experiments older than {days_old} days found!")
        return
    
    print(f"🗑️  OLD EXPERIMENTS (older than {days_old} days):")
    for exp_name, creation_time in old_experiments:
        days = (current_time - creation_time) / (24 * 60 * 60)
        print(f"   {exp_name} ({days:.1f} days old)")
    
    response = input(f"\n❓ Delete {len(old_experiments)} old experiments? (y/N): ")
    if response.lower() == 'y':
        for exp_name, _ in old_experiments:
            exp_path = os.path.join(experiments_dir, exp_name)
            import shutil
            shutil.rmtree(exp_path)
            print(f"🗑️  Deleted: {exp_name}")
        print("✅ Cleanup completed!")
    else:
        print("❌ Cleanup cancelled.")


def main():
    """
    Main function with command line interface
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Experiment Manager')
    parser.add_argument('action', choices=['list', 'compare', 'load', 'clean', 'delete'], 
                       help='Action to perform')
    parser.add_argument('--experiments', nargs='+', help='Experiment names for comparison/loading/deletion')
    parser.add_argument('--days', type=int, default=30, help='Days threshold for cleanup')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_experiments()
    
    elif args.action == 'compare':
        compare_experiments(args.experiments)
    
    elif args.action == 'load':
        if not args.experiments:
            print("❌ Please specify experiment name(s) to load")
            return
        for exp_name in args.experiments:
            load_experiment(exp_name)
    
    elif args.action == 'clean':
        clean_old_experiments(args.days)
    
    elif args.action == 'delete':
        if not args.experiments:
            print("❌ Please specify experiment name(s) to delete")
            return
        delete_specific_experiments(args.experiments)


if __name__ == "__main__":
    main() 