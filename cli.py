#!/usr/bin/env python3
"""
Command Line Interface for RLHF Arena.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from scripts.run_experiment import ExperimentRunner
from scripts.benchmark import BenchmarkOrchestrator
from scripts.evaluate import ResultsEvaluator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RLHF Arena: Benchmarking frontier post-training RL methods for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single PPO experiment
  rlhf-arena run --config configs/ppo.yaml --output experiments/ppo_test
  
  # Run full benchmark suite
  rlhf-arena benchmark --config configs/benchmark.yaml --output experiments/benchmark
  
  # Evaluate results
  rlhf-arena evaluate --results experiments/ --output reports/
  
  # Quick start example
  rlhf-arena quick-start
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run a single experiment')
    run_parser.add_argument('--config', required=True, help='Path to configuration file')
    run_parser.add_argument('--output', required=True, help='Output directory for results')
    run_parser.add_argument('--resume', help='Resume from checkpoint')
    run_parser.add_argument('--algorithm', help='Override algorithm from config')
    run_parser.add_argument('--dataset', help='Override dataset from config')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run full benchmark suite')
    benchmark_parser.add_argument('--config', required=True, help='Path to benchmark configuration')
    benchmark_parser.add_argument('--output', required=True, help='Output directory for results')
    benchmark_parser.add_argument('--max-parallel', type=int, default=1, help='Maximum parallel experiments')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate experiment results')
    evaluate_parser.add_argument('--results', required=True, help='Directory containing experiment results')
    evaluate_parser.add_argument('--output', required=True, help='Output directory for evaluation results')
    evaluate_parser.add_argument('--compare', nargs='+', help='Specific algorithms to compare')
    evaluate_parser.add_argument('--metrics', nargs='+', help='Specific metrics to analyze')
    
    # Quick start command
    quick_start_parser = subparsers.add_parser('quick-start', help='Run quick start example')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'run':
            # Run single experiment
            runner = ExperimentRunner(
                config_path=args.config,
                output_dir=args.output,
                resume_from=args.resume
            )
            
            # Override config if specified
            if args.algorithm:
                runner.config['algorithm'] = args.algorithm
            if args.dataset:
                runner.config['dataset']['name'] = args.dataset
            
            runner.run()
            
        elif args.command == 'benchmark':
            # Run benchmark suite
            orchestrator = BenchmarkOrchestrator(
                config_path=args.config,
                output_dir=args.output,
                max_parallel=args.max_parallel
            )
            orchestrator.run()
            
        elif args.command == 'evaluate':
            # Evaluate results
            evaluator = ResultsEvaluator(
                results_dir=args.results,
                output_dir=args.output
            )
            evaluator.run_evaluation()
            
        elif args.command == 'quick-start':
            # Run quick start example
            from examples.quick_start import main as quick_start_main
            quick_start_main()
            
        elif args.command == 'info':
            # Show system information
            show_system_info()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def show_system_info():
    """Show system information."""
    print("RLHF Arena System Information")
    print("=" * 40)
    
    # Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Check key dependencies
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("PyTorch: Not installed")
    
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")
    
    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
    except ImportError:
        print("Datasets: Not installed")
    
    # Check configuration files
    config_dir = Path("configs")
    if config_dir.exists():
        print(f"\nConfiguration files:")
        for config_file in config_dir.glob("*.yaml"):
            print(f"  ✓ {config_file.name}")
    else:
        print("\nConfiguration files: Not found")
    
    # Check scripts
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        print(f"\nScripts:")
        for script_file in scripts_dir.glob("*.py"):
            print(f"  ✓ {script_file.name}")
    else:
        print("\nScripts: Not found")
    
    # Check examples
    examples_dir = Path("examples")
    if examples_dir.exists():
        print(f"\nExamples:")
        for example_file in examples_dir.glob("*.py"):
            print(f"  ✓ {example_file.name}")
    else:
        print("\nExamples: Not found")


if __name__ == '__main__':
    main() 