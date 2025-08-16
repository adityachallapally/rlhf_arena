#!/usr/bin/env python3
"""
Test utilities for RLHF Arena.
"""

import unittest
import tempfile
import os
import yaml
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rlhf_arena.utils import load_config, setup_logging, setup_device


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_config(self):
        """Test configuration loading."""
        # Create a test config file
        test_config = {
            'test_section': {
                'test_key': 'test_value'
            }
        }
        
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Test loading
        loaded_config = load_config(config_file)
        self.assertEqual(loaded_config, test_config)
    
    def test_setup_logging(self):
        """Test logging setup."""
        log_config = {
            'output_dir': self.temp_dir,
            'log_level': 'INFO'
        }
        
        logger = setup_logging(log_config)
        self.assertIsNotNone(logger)
        self.assertEqual(logger.level, 20)  # INFO level
        
        # Check if log file was created
        log_files = list(Path(self.temp_dir).glob('*.log'))
        self.assertGreater(len(log_files), 0)
    
    def test_setup_device(self):
        """Test device setup."""
        # Test auto device selection
        hardware_config = {'device': 'auto'}
        device = setup_device(hardware_config)
        self.assertIsNotNone(device)
        
        # Test CPU device
        hardware_config = {'device': 'cpu'}
        device = setup_device(hardware_config)
        self.assertEqual(device.type, 'cpu')


if __name__ == '__main__':
    unittest.main() 