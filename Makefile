# RLHF Arena Makefile
# Common development and deployment tasks

.PHONY: help install test clean build docker-build docker-run benchmark evaluate setup

# Default target
help:
	@echo "RLHF Arena - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup          - Install dependencies and setup environment"
	@echo "  install        - Install Python dependencies"
	@echo "  docker-build   - Build Docker image"
	@echo ""
	@echo "Development:"
	@echo "  test           - Run test suite"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code with black"
	@echo ""
	@echo "Experiments:"
	@echo "  run-ppo        - Run PPO experiment"
	@echo "  run-dpo        - Run DPO experiment"
	@echo "  run-grpo       - Run GRPO experiment"
	@echo "  benchmark      - Run full benchmark suite"
	@echo "  evaluate       - Evaluate experiment results"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          - Clean build artifacts and cache"
	@echo "  logs           - View experiment logs"
	@echo "  status         - Check experiment status"

# Setup environment
setup: install
	@echo "Setting up RLHF Arena environment..."
	@mkdir -p experiments reports checkpoints logs
	@echo "Environment setup complete!"

# Install dependencies
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

# Build Docker image
docker-build:
	@echo "Building Docker image..."
	docker build -t rlhf_arena .
	@echo "Docker image built successfully!"

# Run Docker container
docker-run:
	@echo "Running Docker container..."
	docker run -it --gpus all -v $(PWD):/workspace rlhf_arena

# Run tests
test:
	@echo "Running test suite..."
	python -m pytest tests/ -v --tb=short

# Code linting
lint:
	@echo "Running code linting..."
	flake8 rlhf_arena/ scripts/ tests/
	mypy rlhf_arena/ scripts/

# Code formatting
format:
	@echo "Formatting code..."
	black rlhf_arena/ scripts/ tests/
	isort rlhf_arena/ scripts/ tests/

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
	@echo "Cleanup complete!"

# Run PPO experiment
run-ppo:
	@echo "Running PPO experiment..."
	python scripts/run_experiment.py \
		--config configs/ppo.yaml \
		--output_dir experiments/ppo_experiment

# Run DPO experiment
run-dpo:
	@echo "Running DPO experiment..."
	python scripts/run_experiment.py \
		--config configs/dpo.yaml \
		--output_dir experiments/dpo_experiment

# Run GRPO experiment
run-grpo:
	@echo "Running GRPO experiment..."
	python scripts/run_experiment.py \
		--config configs/grpo.yaml \
		--output_dir experiments/grpo_experiment

# Run full benchmark
benchmark:
	@echo "Running full benchmark suite..."
	python scripts/benchmark.py \
		--config configs/multiobjective.yaml \
		--output_dir experiments/benchmark \
		--max_parallel 2

# Evaluate results
evaluate:
	@echo "Evaluating experiment results..."
	python scripts/evaluate.py \
		--results_dir experiments/ \
		--output_dir reports/evaluation

# View experiment logs
logs:
	@echo "Recent experiment logs:"
	@find experiments/ -name "*.log" -exec tail -n 20 {} \; 2>/dev/null || echo "No logs found"

# Check experiment status
status:
	@echo "Experiment Status:"
	@echo "=================="
	@find experiments/ -maxdepth 1 -type d -name "*_experiment" | while read dir; do \
		echo "$$(basename $$dir): $$(ls -la $$dir/*.log 2>/dev/null | wc -l) log files"; \
	done
	@echo ""
	@echo "Benchmark Status:"
	@echo "================="
	@if [ -f experiments/benchmark/benchmark_state.json ]; then \
		echo "Benchmark state file found"; \
		python -c "import json; data=json.load(open('experiments/benchmark/benchmark_state.json')); print(f'Completed: {len(data.get(\"completed\", []))}, Failed: {len(data.get(\"failed\", []))}')"; \
	else \
		echo "No benchmark state found"; \
	fi

# Quick start - run basic experiments
quick-start: run-ppo run-dpo run-grpo
	@echo "Quick start experiments completed!"

# Full pipeline - run everything
full-pipeline: benchmark evaluate
	@echo "Full pipeline completed!"

# Development mode - install in editable mode
dev-install:
	@echo "Installing in development mode..."
	pip install -e .
	@echo "Development installation complete!"

# Generate documentation
docs:
	@echo "Generating documentation..."
	pandoc README.md -o README.pdf
	@echo "Documentation generated!"

# Performance profiling
profile:
	@echo "Running performance profiling..."
	python -m cProfile -o profile_output.prof scripts/run_experiment.py \
		--config configs/ppo.yaml \
		--output_dir experiments/profile_test

# Memory profiling
memory-profile:
	@echo "Running memory profiling..."
	python -m memory_profiler scripts/run_experiment.py \
		--config configs/ppo.yaml \
		--output_dir experiments/memory_test

# GPU monitoring
gpu-monitor:
	@echo "GPU monitoring (requires nvidia-smi)..."
	watch -n 1 nvidia-smi

# System info
sys-info:
	@echo "System Information:"
	@echo "==================="
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "GPU Info:"; \
		nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits; \
	else \
		echo "nvidia-smi not available"; \
	fi

# Backup experiments
backup:
	@echo "Creating backup of experiments..."
	tar -czf experiments_backup_$$(date +%Y%m%d_%H%M%S).tar.gz experiments/
	@echo "Backup created!"

# Restore from backup
restore:
	@echo "Available backups:"
	@ls -la experiments_backup_*.tar.gz 2>/dev/null || echo "No backups found"
	@echo ""
	@echo "To restore, use: make restore-backup BACKUP_FILE=filename.tar.gz"

# Restore specific backup
restore-backup:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore-backup BACKUP_FILE=filename.tar.gz"; \
		exit 1; \
	fi
	@echo "Restoring from $(BACKUP_FILE)..."
	tar -xzf $(BACKUP_FILE)
	@echo "Restore complete!"

# Health check
health-check:
	@echo "Running health checks..."
	@echo "1. Checking Python environment..."
	@python -c "import torch, transformers, datasets; print('✓ Core dependencies OK')"
	@echo "2. Checking configuration files..."
	@ls -la configs/*.yaml >/dev/null && echo "✓ Configuration files OK" || echo "✗ Configuration files missing"
	@echo "3. Checking scripts..."
	@ls -la scripts/*.py >/dev/null && echo "✓ Scripts OK" || echo "✗ Scripts missing"
	@echo "4. Checking output directories..."
	@mkdir -p experiments reports checkpoints logs
	@echo "✓ Output directories OK"
	@echo "Health check complete!"

# Update dependencies
update-deps:
	@echo "Updating dependencies..."
	pip install --upgrade -r requirements.txt
	@echo "Dependencies updated!"

# Create new experiment config
new-config:
	@if [ -z "$(ALGORITHM)" ]; then \
		echo "Usage: make new-config ALGORITHM=algorithm_name"; \
		echo "Available algorithms: ppo, dpo, grpo, grpo_offpolicy, grpo_vi, rlaif"; \
		exit 1; \
	fi
	@echo "Creating new config for $(ALGORITHM)..."
	@cp configs/$(ALGORITHM).yaml configs/$(ALGORITHM)_custom.yaml
	@echo "Custom config created: configs/$(ALGORITHM)_custom.yaml"

# Run custom experiment
run-custom:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Usage: make run-custom CONFIG=config_path OUTPUT=output_dir"; \
		exit 1; \
	fi
	@echo "Running custom experiment with $(CONFIG)..."
	python scripts/run_experiment.py \
		--config $(CONFIG) \
		--output_dir $(OUTPUT)

# Monitor experiments
monitor:
	@echo "Monitoring active experiments..."
	@ps aux | grep "run_experiment.py\|benchmark.py" | grep -v grep || echo "No experiments running"
	@echo ""
	@echo "GPU usage:"
	@nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "GPU monitoring not available"

# Stop all experiments
stop-all:
	@echo "Stopping all experiments..."
	@pkill -f "run_experiment.py\|benchmark.py" || echo "No experiments to stop"
	@echo "All experiments stopped!"

# Generate experiment report
report:
	@echo "Generating experiment report..."
	@if [ -d "experiments/" ]; then \
		python scripts/evaluate.py --results_dir experiments/ --output_dir reports/; \
		echo "Report generated in reports/"; \
	else \
		echo "No experiments directory found"; \
	fi

# Archive completed experiments
archive:
	@echo "Archiving completed experiments..."
	@if [ -d "experiments/" ]; then \
		mkdir -p archives/$$(date +%Y%m); \
		find experiments/ -name "experiment_summary.json" -exec dirname {} \; | \
		while read dir; do \
			exp_name=$$(basename $$dir); \
			tar -czf "archives/$$(date +%Y%m)/$${exp_name}_$$(date +%Y%m%d).tar.gz" -C $$(dirname $$dir) $$exp_name; \
		done; \
		echo "Archives created in archives/$$(date +%Y%m)/"; \
	else \
		echo "No experiments directory found"; \
	fi
