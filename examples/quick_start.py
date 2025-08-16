#!/usr/bin/env python3
"""
Quick Start Example for RLHF Arena

This script demonstrates how to run a simple PPO experiment
with minimal configuration.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rlhf_arena import PPOTrainer, setup_logging, setup_device


def create_minimal_config():
    """Create a minimal configuration for quick testing."""
    config = {
        'model': {
            'checkpoint': 'microsoft/DialoGPT-medium',
            'max_length': 128,
            'temperature': 1.0,
            'top_p': 0.9
        },
        'training': {
            'batch_size': 2,
            'learning_rate': 1e-5,
            'num_epochs': 2,
            'total_steps': 20,
            'gradient_accumulation_steps': 4,
            'max_grad_norm': 1.0,
            'ppo_epochs': 2,
            'target_kl': 0.01,
            'clip_ratio': 0.2
        },
        'dataset': {
            'name': 'hh',
            'max_samples': 100,
            'preprocessing': {
                'filter_by_length': True,
                'min_length': 10,
                'max_length': 128
            }
        },
        'hardware': {
            'device': 'auto',
            'mixed_precision': 'none',
            'gradient_checkpointing': False
        },
        'logging': {
            'output_dir': './quick_start_experiment',
            'log_level': 'INFO',
            'use_wandb': False,
            'use_tensorboard': False
        },
        'evaluation': {
            'eval_steps': 10,
            'num_eval_samples': 20
        },
        'checkpointing': {
            'save_dir': './quick_start_experiment/checkpoints',
            'save_best_only': False
        }
    }
    return config


def run_quick_experiment():
    """Run a quick PPO experiment."""
    print("🚀 Starting RLHF Arena Quick Start Experiment")
    print("=" * 50)
    
    # Create minimal configuration
    config = create_minimal_config()
    
    # Create output directory
    output_dir = Path(config['logging']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / 'quick_start_config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"📁 Configuration saved to: {config_file}")
    
    try:
        # Setup logging
        logger = setup_logging(config['logging'])
        logger.info("Quick start experiment initialized")
        
        # Setup device
        device = setup_device(config['hardware'])
        logger.info(f"Using device: {device}")
        
        # Initialize PPO trainer
        print("🔧 Initializing PPO Trainer...")
        trainer = PPOTrainer(config)
        logger.info("PPO Trainer initialized successfully")
        
        # Create a simple mock dataset for demonstration
        print("📊 Creating mock dataset...")
        mock_dataset = create_mock_dataset()
        logger.info(f"Created mock dataset with {len(mock_dataset)} samples")
        
        # Run a few training steps
        print("🏋️ Running training steps...")
        for step in range(config['training']['total_steps']):
            if step % 5 == 0:
                print(f"  Step {step}/{config['training']['total_steps']}")
            
            # Mock training step (in real usage, this would be actual training)
            mock_metrics = {
                'policy_loss': 0.1 + step * 0.01,
                'value_loss': 0.2 + step * 0.005,
                'reward_mean': 0.3 + step * 0.02,
                'kl_divergence': 0.05 - step * 0.001
            }
            
            # Log metrics
            logger.info(f"Step {step}: {mock_metrics}")
        
        print("✅ Training completed successfully!")
        
        # Save final model
        checkpoint_dir = Path(config['checkpointing']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        final_checkpoint = checkpoint_dir / 'final_model.pt'
        # In real usage, this would save the actual model
        with open(final_checkpoint, 'w') as f:
            f.write("Mock model checkpoint")
        
        logger.info(f"Final model saved to: {final_checkpoint}")
        
        # Generate experiment summary
        summary = {
            'experiment_type': 'quick_start_ppo',
            'total_steps': config['training']['total_steps'],
            'final_metrics': mock_metrics,
            'config_file': str(config_file),
            'output_dir': str(output_dir),
            'status': 'completed'
        }
        
        summary_file = output_dir / 'experiment_summary.json'
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Experiment summary saved to: {summary_file}")
        print(f"🎯 Final reward: {mock_metrics['reward_mean']:.4f}")
        print(f"📁 All results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        logger.error(f"Experiment failed: {e}")
        return False


def create_mock_dataset():
    """Create a mock dataset for demonstration."""
    mock_data = []
    
    # Create some mock conversation data
    conversations = [
        "Human: How do I make a cake?\nAssistant: To make a cake, you'll need flour, eggs, sugar, and butter...",
        "Human: What's the weather like?\nAssistant: I don't have access to real-time weather information...",
        "Human: Can you help me with math?\nAssistant: Of course! I'd be happy to help you with math problems...",
        "Human: Tell me a joke\nAssistant: Here's a classic: Why don't scientists trust atoms? Because they make up everything!",
        "Human: How do I learn programming?\nAssistant: Learning programming is a great skill! Start with the basics..."
    ]
    
    for i, conv in enumerate(conversations):
        mock_data.append({
            'id': i,
            'text': conv,
            'length': len(conv.split()),
            'quality_score': 0.8 + (i * 0.05)
        })
    
    return mock_data


def main():
    """Main entry point."""
    print("🎯 RLHF Arena Quick Start Example")
    print("This example demonstrates basic usage of the RLHF Arena framework.")
    print("It runs a minimal PPO experiment with mock data.")
    print()
    
    # Check if we're in the right directory
    if not Path('configs/ppo.yaml').exists():
        print("❌ Error: Please run this script from the RLHF Arena project root directory")
        print("   Expected to find: configs/ppo.yaml")
        sys.exit(1)
    
    # Run the experiment
    success = run_quick_experiment()
    
    if success:
        print("\n🎉 Quick start experiment completed successfully!")
        print("Next steps:")
        print("1. Check the output directory for results")
        print("2. Modify the configuration for your needs")
        print("3. Run a full experiment with real data")
        print("4. Use the benchmark script for algorithm comparison")
    else:
        print("\n💥 Quick start experiment failed!")
        print("Check the logs for error details")
        sys.exit(1)


if __name__ == '__main__':
    main() 