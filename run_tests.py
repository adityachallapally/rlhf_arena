#!/usr/bin/env python3
"""
Simple test runner for RLHF Arena.
"""

import sys
import os
import unittest
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def run_tests():
    """Run all tests."""
    print("Running RLHF Arena tests...")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / 'tests'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code) 