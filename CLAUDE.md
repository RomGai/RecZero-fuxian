# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Logic-RL is a research project implementing reinforcement learning from human feedback (RLHF) for improving LLM reasoning on logic puzzles. Built on top of the veRL framework (ByteDance's Volcano Engine RL for LLM), it specifically focuses on Knights and Knaves puzzles using rule-based rewards with GRPO (Group Relative Policy Optimization).

## Tech Stack

- **Python 3.9+** with PyTorch 2.4.0
- **vLLM 0.6.3** for efficient LLM inference
- **Ray** for distributed computing
- **Hydra** for configuration management
- **FSDP** (Fully Sharded Data Parallel) for multi-GPU training
- **Flash Attention** for optimized attention mechanisms

## Essential Commands

### Installation
```bash
# Create conda environment
conda create -n logic python=3.9
conda activate logic

# Install PyTorch and core dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install vllm==0.6.3 ray
pip3 install flash-attn --no-build-isolation

# Install the verl package in development mode
pip install -e .

# Install additional dependencies
pip install wandb IPython matplotlib
```

### Training
```bash
bash main_grpo.sh      # Main GRPO training (requires 4×A100 80G)
bash main_reinforce.sh # REINFORCE training alternative

# Alternative: Use specific example scripts
bash examples/grpo_trainer/run_qwen2-7b.sh
bash examples/ppo_trainer/run_deepseek7b_llm.sh
```

### Code Formatting
```bash
bash scripts/format.sh  # Uses yapf for Python formatting (formats verl/, tests/, examples/)
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/gpu_utility/
python -m pytest tests/model/
python -m pytest tests/ray/
```

### Evaluation
```bash
bash eval_kk/eval.sh   # Knights and Knaves puzzle evaluation

# Manual evaluation with specific models
python eval_kk/main_eval_instruct.py --model MODEL_PATH --eval_nppl 2
```

### Documentation
```bash
cd docs
pip install -r requirements-docs.txt
make clean && make html

# Serve documentation locally
python -m http.server -d _build/html/
# Open localhost:8000 in browser
```

### Data Preprocessing
```bash
# For base models
python examples/data_preprocess/kk.py --local_dir OUTPUT_PATH --data_path INPUT_PATH

# For instruct models  
python examples/data_preprocess/kk.py --template_type=qwen-instruct --local_dir OUTPUT_PATH --data_path INPUT_PATH
```

## Architecture Overview

### Core Package Structure (`/verl/`)

**Distributed Training Pipeline**:
- `trainer/` - Training orchestration with `main_ppo.py` as primary entry point
- `workers/` - Ray-based distributed workers:
  - `actor/` - Policy model workers for action sampling
  - `critic/` - Value model workers for advantage estimation  
  - `rollout/` - Experience generation and collection
  - `reward_model/` - Reward scoring and feedback
- `single_controller/` - Ray cluster coordination and resource management

**Key Components**:
- `models/` - Optimized model implementations with FSDP support
- `utils/reward_score/kk.py` - Knights & Knaves logic puzzle scoring engine
- `trainer/config/` - Hydra YAML configurations for different training setups
- `trainer/ppo/ray_trainer.py` - Core distributed PPO implementation

### Configuration System

Uses Hydra for hierarchical configuration management. Main configs in `/verl/trainer/config/`:
- `ppo_trainer.yaml` - Primary training configuration
- Environment variables control vLLM backend settings
- Modular reward function selection through config files

### Training Flow

1. **Ray Cluster Setup** - Distributed worker initialization
2. **Model Loading** - Actor/critic models with FSDP sharding
3. **Rollout Generation** - Experience collection from environment
4. **Reward Scoring** - Rule-based evaluation using logic solvers
5. **PPO Updates** - Policy optimization with advantage estimation
6. **Evaluation** - Periodic assessment on held-out puzzles

## Important Implementation Details

### Reward System
The project implements rule-based rewards for Knights and Knaves puzzles in `/verl/utils/reward_score/kk.py`. This module contains the core logic for validating puzzle solutions and assigning rewards based on correctness.

### Memory Management
Training requires significant GPU memory (4×A100 80G recommended). The codebase uses FSDP for model sharding and vLLM's optimized attention mechanisms for efficient inference during rollouts.

### Hybrid Engine Pattern
The architecture follows an actor-rollout-ref pattern where different worker types handle specific aspects of the RL pipeline, allowing for efficient resource utilization and scaling.

## Data and Evaluation

- Training data located in `/data/` includes Amazon rating datasets and Knights & Knaves puzzles
- Evaluation scripts focus on logic puzzle performance metrics
- Comprehensive test coverage in `/tests/` for core functionality

## Development Notes

### Environment Variables
Key environment variables for training:
- `VLLM_ATTENTION_BACKEND=XFORMERS` - Use XFormers attention backend
- `CUDA_VISIBLE_DEVICES` - Control GPU visibility (e.g., `0,1,2,3`)
- `MASTER_PORT` - Ray cluster master port (default: 123456)
- `WANDB_MODE=offline` - Run W&B in offline mode
- `MLFLOW_TRACKING_URI` - MLflow tracking URI for experiment logging
- `TOKENIZERS_PARALLELISM=true` - Enable tokenizer parallelism

### Training Requirements
- **Minimum Hardware**: 4×A100 80G GPUs for full training
- **Memory Usage**: Requires ~240GB GPU memory total with FSDP offloading
- **Ray Cluster**: Distributed training requires Ray initialization
- **Model Paths**: Update MODEL_PATH variables in training scripts before running

### Configuration Files
Training behavior is controlled through Hydra YAML configs in `verl/trainer/config/`:
- Modify hyperparameters, data paths, and model settings
- Override config values via command line: `algorithm.kl_ctrl.kl_coef=0.001`
- Environment-specific settings can be templated using `${oc.env:VAR_NAME}`

### Code Standards
- Code follows yapf formatting standards with config in `.style.yapf` (if present)
- Apache 2.0 licensed with ByteDance copyright headers
- Built for research use with production-ready distributed training capabilities
- Configuration-driven approach allows easy experimentation with different hyperparameters and reward functions