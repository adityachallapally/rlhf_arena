#!/usr/bin/env python3
"""
Setup script for RLHF Arena.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    with open(this_directory / "requirements.txt", "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="rlhf_arena",
    version="0.1.0",
    author="RLHF Arena Contributors",
    author_email="support@rlhf-arena.org",
    description="A comprehensive benchmarking framework for Reinforcement Learning from Human Feedback (RLHF) algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/rlhf_arena",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "isort>=5.12.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "full": [
            "flash-attn>=2.0.0",
            "xformers>=0.0.20",
            "bitsandbytes>=0.41.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rlhf-arena=rlhf_arena.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rlhf_arena": ["configs/*.yaml", "scripts/*.py"],
    },
    zip_safe=False,
) 