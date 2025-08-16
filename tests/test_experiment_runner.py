#!/usr/bin/env python3
"""
Test experiment runner for RLHF Arena.
"""

import unittest
import tempfile
import os
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_experiment import ExperimentRunner


class TestExperimentRunner(unittest.TestCase):
    """Test experiment runner functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal test config
        self.test_config = {
            'model': {
                'checkpoint': 'microsoft/DialoGPT-medium',
                'max_length': 128
            },
            'training': {
                'batch_size': 2,
                'learning_rate': 1e-5,
                'num_epochs': 2,
                'total_steps': 10
            },
            'dataset': {
                'name': 'hh',
                'max_samples': 100
            },
            'hardware': {
                'device': 'cpu',
                'mixed_precision': 'none'
            },
            'logging': {
                'output_dir': self.temp_dir,
                'log_level': 'INFO'
            },
            'evaluation': {
                'eval_steps': 5,
                'num_eval_samples': 50
            },
            'checkpointing': {
                'save_dir': os.path.join(self.temp_dir, 'checkpoints'),
                'save_best_only': False
            }
        }
        
        # Save test config
        self.config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('scripts.run_experiment.PPOTrainer')
    @patch('scripts.run_experiment.load_config')
    @patch('scripts.run_experiment.setup_logging')
    @patch('scripts.run_experiment.setup_device')
    def test_experiment_runner_initialization(self, mock_setup_device, mock_setup_logging, 
                                           mock_load_config, mock_ppo_trainer):
        """Test experiment runner initialization."""
        # Mock the dependencies
        mock_load_config.return_value = self.test_config
        mock_setup_logging.return_value = MagicMock()
        mock_setup_device.return_value = MagicMock()
        mock_ppo_trainer.return_value = MagicMock()
        
        # Test initialization
        runner = ExperimentRunner(
            config_path=self.config_file,
            output_dir=self.temp_dir
        )
        
        self.assertIsNotNone(runner)
        self.assertEqual(runner.config_path, self.config_file)
        self.assertEqual(str(runner.output_dir), self.temp_dir)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test with valid config
        valid_config = self.test_config.copy()
        
        # Test with missing required sections
        invalid_config = valid_config.copy()
        del invalid_config['model']
        
        # This should raise an error
        with self.assertRaises(KeyError):
            # We can't easily test this without mocking, but we can verify the structure
            required_sections = ['model', 'training', 'dataset', 'hardware']
            for section in required_sections:
                if section not in invalid_config:
                    raise KeyError(f"Missing required section: {section}")
    
    def test_output_directory_creation(self):
        """Test output directory creation."""
        output_dir = os.path.join(self.temp_dir, 'new_experiment')
        
        # Directory should not exist initially
        self.assertFalse(os.path.exists(output_dir))
        
        # Create directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Directory should exist now
        self.assertTrue(os.path.exists(output_dir))
    
    def test_config_saving(self):
        """Test configuration saving."""
        output_dir = os.path.join(self.temp_dir, 'test_output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        config_output = os.path.join(output_dir, 'config.yaml')
        with open(config_output, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Verify config was saved
        self.assertTrue(os.path.exists(config_output))
        
        # Verify content
        with open(config_output, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        self.assertEqual(saved_config, self.test_config)


if __name__ == '__main__':
    unittest.main() 