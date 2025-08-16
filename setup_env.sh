#!/bin/bash
# RLHF Arena Environment Setup Script
# This script sets up the environment for RLHF Arena

set -e  # Exit on any error

echo "🚀 Setting up RLHF Arena Environment"
echo "====================================="

# Check if Python 3.8+ is available
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ is required, but found Python $python_version"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

echo "✅ Python $python_version detected"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not available"
    echo "Please install pip3 and try again"
    exit 1
fi

echo "✅ pip3 detected"

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3 -m venv rlhf_arena_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source rlhf_arena_env/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Install package in development mode
echo "🔧 Installing RLHF Arena in development mode..."
pip install -e .

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p experiments reports checkpoints logs cache archives exports

# Set permissions
echo "🔐 Setting permissions..."
chmod +x scripts/*.py
chmod +x examples/*.py

echo ""
echo "🎉 Environment setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source rlhf_arena_env/bin/activate"
echo ""
echo "To run tests:"
echo "  python run_tests.py"
echo ""
echo "To run a quick start example:"
echo "  python examples/quick_start.py"
echo ""
echo "To check system information:"
echo "  python cli.py info"
echo ""
echo "For more information, see README.md" 