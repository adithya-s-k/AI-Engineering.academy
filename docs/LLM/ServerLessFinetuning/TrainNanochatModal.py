"""
Train nanochat on Modal: Build Your Own ChatGPT from Scratch

This Modal script runs the complete nanochat pipeline - from tokenizer training
to a fully functional ChatGPT clone. nanochat is Andrej Karpathy's minimal,
educational implementation of a full-stack LLM training pipeline.

The pipeline includes:
1. Tokenizer Training - Custom BPE tokenizer (65K vocab)
2. Base Model Pretraining - Train GPT on FineWeb dataset
3. Midtraining - Teach conversation tokens and tool use
4. Supervised Fine-tuning - Task-specific training
5. Reinforcement Learning - Optional RL on GSM8K
6. Comprehensive Evaluation - CORE, ARC, GSM8K, HumanEval, MMLU
7. Inference - Chat CLI and Web UI

GPU Requirements:
- Recommended: 4-8x A100 80GB for full speedrun (~4 hours, ~$96)
- Minimum: 1x A100 80GB (will take 8x longer)
- Testing: 1x A100 40GB (with reduced batch sizes)

Usage:
    # Full speedrun pipeline (~4 hours on 8xA100)
    modal run TrainNanochatModal.py

    # Quick test mode (smaller model, less data)
    modal run TrainNanochatModal.py::main --num-data-shards=8 --depth=12

    # Run individual stages:
    modal run TrainNanochatModal.py::download_dataset --num-shards=240
    modal run TrainNanochatModal.py::train_tokenizer
    modal run TrainNanochatModal.py::train_base_model --depth=20
    modal run TrainNanochatModal.py::train_mid_model
    modal run TrainNanochatModal.py::train_sft_model
    modal run TrainNanochatModal.py::chat_cli --source=sft --prompt="Why is the sky blue?"

    # Launch web UI
    modal run TrainNanochatModal.py::chat_web --source=sft

For more information: https://github.com/karpathy/nanochat
"""

from modal import App, Image as ModalImage, Volume

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Time constants
MINUTES = 60
HOURS = 60 * 60

# GPU Configuration
# Options: "a100-80gb", "a100-40gb", "a10g", "l40s", "h100", etc.
GPU_TYPE = "a100-80gb"

# Multi-GPU configuration for different stages
# Nanochat is designed for 8 GPUs but works with 1-8
NUM_GPUS_BASE = 4      # Base model training (can be 1-8)
NUM_GPUS_MID = 4       # Midtraining (can be 1-8)
NUM_GPUS_SFT = 4       # Supervised fine-tuning (can be 1-8)
NUM_GPUS_RL = 4        # Reinforcement learning (can be 1-8)
NUM_GPUS_EVAL = 4      # Evaluation (can be 1-8)
NUM_GPUS_TOKENIZER = 1 # Tokenizer training (single GPU)
NUM_GPUS_INFERENCE = 1 # Inference (single GPU)

# Training Configuration
WANDB_PROJECT_DEFAULT = "nanochat-modal"
BASE_DIR = "/root/.cache/nanochat"  # Cache directory inside container

# =============================================================================
# MODAL APP AND VOLUME SETUP
# =============================================================================

app = App("nanochat-training")

# Create volumes for persistent storage
# Volume 1: Dataset (FineWeb shards)
data_volume = Volume.from_name("nanochat-data", create_if_missing=True)

# Volume 2: Checkpoints and outputs
checkpoint_volume = Volume.from_name("nanochat-checkpoints", create_if_missing=True)

# Configure volume mounting points
VOLUME_CONFIG = {
    "/data": data_volume,              # Dataset storage
    "/checkpoints": checkpoint_volume, # Model checkpoints
}

# =============================================================================
# CONTAINER IMAGE SETUP
# =============================================================================

# Build a comprehensive image with everything nanochat needs:
# - CUDA 12.8 for GPU support
# - Python 3.11
# - Rust/Cargo for building the custom tokenizer
# - All Python dependencies
# - The nanochat codebase
# - Compiled Rust tokenizer (rustbpe)

NANOCHAT_IMAGE = (
    # Start with NVIDIA CUDA image for GPU support
    ModalImage.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04",
        add_python="3.11"
    )

    # Install system dependencies
    .apt_install(
        "git",
        "build-essential",
        "curl",
        "wget",
        "unzip",
    )

    # Install Rust and Cargo (needed for building the tokenizer)
    # The tokenizer is implemented in Rust for speed
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> $HOME/.bashrc",
    )

    # Install Python dependencies
    # These are all the packages nanochat needs
    .pip_install(
        "torch>=2.8.0",              # PyTorch for deep learning
        "datasets>=4.0.0",           # HuggingFace datasets
        "wandb>=0.21.3",             # Experiment tracking (optional)
        "fastapi>=0.117.1",          # Web server for chat UI
        "uvicorn>=0.36.0",           # ASGI server
        "numpy==1.26.4",             # Numerical computing
        "tiktoken>=0.11.0",          # OpenAI's tokenizer (for reference)
        "tokenizers>=0.22.0",        # HuggingFace tokenizers
        "regex>=2025.9.1",           # Regular expressions
        "psutil>=7.1.0",             # System utilities
        "pandas",                    # Data manipulation
        "pyyaml",                    # YAML config files
        "pyarrow",                   # Apache Arrow (for parquet files)
        "requests",                  # HTTP requests
        "maturin>=1.9.4",           # Build Rust Python extensions
    )

    # Copy the entire nanochat directory from local filesystem into the image
    # Make sure you have cloned nanochat in the same directory as this script!
    # git clone https://github.com/karpathy/nanochat.git
    .add_local_dir(
        local_path="nanochat",
        remote_path="/root/nanochat",
        copy=True
    )

    # Set working directory
    .workdir("/root/nanochat")

    # Build the Rust tokenizer extension
    # This compiles the rustbpe package and installs it
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && maturin develop --release --manifest-path rustbpe/Cargo.toml'",
    )

    # Set environment variables
    .env({
        "OMP_NUM_THREADS": "1",  # OpenMP threading
        "NANOCHAT_BASE_DIR": BASE_DIR,
        "HF_HOME": "/data/.cache/huggingface",  # HuggingFace cache
    })
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_base_dir():
    """Create base directory structure for nanochat artifacts."""
    import os
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/base_data", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/tokenizer", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/eval_bundle", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/report", exist_ok=True)

def run_torchrun_command(
    script: str,
    num_gpus: int,
    extra_args: list = None
):
    """
    Helper to run a nanochat script with torchrun for multi-GPU training.

    Args:
        script: Python module to run (e.g., "scripts.base_train")
        num_gpus: Number of GPUs to use
        extra_args: Additional command-line arguments
    """
    import subprocess

    if extra_args is None:
        extra_args = []

    # Build the torchrun command
    # --standalone: Single-node training
    # --nproc_per_node: Number of processes (one per GPU)
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        "-m",
        script,
    ]

    # Add extra args after "--" separator
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")

    return result

# =============================================================================
# STAGE 1: DATASET DOWNLOAD
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,  # Dataset download can take a while
    # No GPU needed for downloading - saves money!
)
def download_dataset(num_shards: int = 240):
    """
    Download FineWeb dataset shards from HuggingFace.

    The nanochat pretraining dataset is FineWeb-edu-100B, split into 1822 shards.
    Each shard is ~250M characters (~100MB compressed).

    For the speedrun (d20 model with 561M params):
    - Chinchilla ratio: 20x tokens = 11.2B tokens needed
    - At 4.8 chars/token: ~54B characters needed
    - Number of shards: 54B / 250M = 216, rounded to 240 for safety

    For testing: Use 8 shards (~2B characters)

    Args:
        num_shards: Number of shards to download (8 for testing, 240 for full speedrun)
    """
    import subprocess

    setup_base_dir()

    print("=" * 80)
    print(f"DOWNLOADING FINEWEB DATASET - {num_shards} SHARDS")
    print("=" * 80)
    print("Each shard: ~250M characters (~100MB)")
    print(f"Total data: ~{num_shards * 250 / 1000:.1f}B characters (~{num_shards * 100 / 1024:.1f}GB)")
    print()

    # Run nanochat's dataset download script
    # This downloads parquet files from HuggingFace
    result = subprocess.run(
        ["python", "-m", "nanochat.dataset", "-n", str(num_shards)],
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Dataset download failed with code {result.returncode}")

    # Commit changes to the volume
    data_volume.commit()

    print("\n" + "=" * 80)
    print(f"✓ Downloaded {num_shards} shards successfully!")
    print("=" * 80)

    return {
        "status": "completed",
        "num_shards": num_shards,
        "data_dir": f"{BASE_DIR}/base_data",
    }

# =============================================================================
# STAGE 2: TOKENIZER TRAINING
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_TOKENIZER}",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def train_tokenizer(
    max_chars: int = 2_000_000_000,  # 2 billion characters
    vocab_size: int = 65536,          # 2^16 = 65536
    doc_cap: int = 10000,             # Max chars per document
):
    """
    Train a custom BPE tokenizer on FineWeb data.

    nanochat uses a custom Rust-based BPE tokenizer for speed and efficiency.
    This is similar to GPT-4's tokenizer design.

    Training on 2B characters takes about 30-60 minutes on a single GPU.

    Args:
        max_chars: Maximum characters to train on (default: 2B)
        vocab_size: Vocabulary size (default: 65536 = 2^16)
        doc_cap: Maximum characters per document (default: 10K)
    """
    import subprocess

    setup_base_dir()

    print("=" * 80)
    print("TRAINING CUSTOM BPE TOKENIZER")
    print("=" * 80)
    print(f"Max characters: {max_chars:,}")
    print(f"Vocabulary size: {vocab_size:,}")
    print(f"Document cap: {doc_cap:,}")
    print()

    # Build training command
    cmd = [
        "python", "-m", "scripts.tok_train",
        f"--max_chars={max_chars}",
        f"--vocab_size={vocab_size}",
        f"--doc_cap={doc_cap}",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer training failed with code {result.returncode}")

    # Commit the trained tokenizer to the volume
    data_volume.commit()
    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("✓ Tokenizer training completed!")
    print(f"Tokenizer saved to {BASE_DIR}/tokenizer/")
    print("=" * 80)

    return {
        "status": "completed",
        "max_chars": max_chars,
        "vocab_size": vocab_size,
        "tokenizer_dir": f"{BASE_DIR}/tokenizer",
    }

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_TOKENIZER}",
    volumes=VOLUME_CONFIG,
    timeout=30 * MINUTES,
)
def evaluate_tokenizer():
    """
    Evaluate the trained tokenizer.

    Reports compression ratio and other metrics.
    """
    import subprocess

    print("=" * 80)
    print("EVALUATING TOKENIZER")
    print("=" * 80)

    result = subprocess.run(
        ["python", "-m", "scripts.tok_eval"],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer evaluation failed with code {result.returncode}")

    print("\n" + "=" * 80)
    print("✓ Tokenizer evaluation completed!")
    print("=" * 80)

    return {"status": "completed"}

# =============================================================================
# STAGE 3: BASE MODEL PRETRAINING
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_BASE}",
    volumes=VOLUME_CONFIG,
    timeout=8 * HOURS,  # Can take 2-4 hours for speedrun
)
def train_base_model(
    depth: int = 20,              # Model depth (20=561M params, 26=~1B params)
    device_batch_size: int = 32,  # Per-device batch size (reduce if OOM)
    max_iterations: int = -1,     # -1 = auto-calculate from Chinchilla ratio
    wandb_run: str = "dummy",     # WandB run name ("dummy" = no logging)
):
    """
    Pretrain the base GPT model on FineWeb.

    This is the main pretraining phase where the model learns language.
    Uses the Muon optimizer for linear layers and AdamW for embeddings.

    Model Architecture (derived from depth):
    - depth=20: 561M parameters (default speedrun)
    - depth=26: ~1B parameters (GPT-2 grade)
    - model_dim = depth * 64
    - num_heads = model_dim / 128

    Training Duration:
    - 8 GPUs: ~2-3 hours for speedrun
    - 1 GPU: ~16-24 hours for speedrun

    Args:
        depth: Model depth (determines size: 20=561M, 26=1B)
        device_batch_size: Batch size per GPU (reduce if OOM)
        max_iterations: Training steps (-1 for auto from Chinchilla)
        wandb_run: WandB run name ("dummy" to disable)
    """
    import subprocess
    import os

    setup_base_dir()

    # Download eval bundle if not present (needed for CORE metric)
    eval_bundle_path = f"{BASE_DIR}/eval_bundle"
    if not os.path.exists(eval_bundle_path):
        print("Downloading eval bundle for CORE metric...")
        subprocess.run([
            "curl", "-L", "-o", "eval_bundle.zip",
            "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
        ], check=True)
        subprocess.run(["unzip", "-q", "eval_bundle.zip"], check=True)
        subprocess.run(["mv", "eval_bundle", eval_bundle_path], check=True)
        subprocess.run(["rm", "eval_bundle.zip"], check=True)

    print("=" * 80)
    print("PRETRAINING BASE MODEL ON FINEWEB")
    print("=" * 80)
    print(f"Model depth: {depth}")
    print(f"Estimated parameters: {depth * depth * 64 * 12 // 1_000_000}M")
    print(f"Device batch size: {device_batch_size}")
    print(f"Number of GPUs: {NUM_GPUS_BASE}")
    print(f"WandB run: {wandb_run}")
    print()

    # Build training command
    extra_args = [
        f"--depth={depth}",
        f"--device_batch_size={device_batch_size}",
        f"--run={wandb_run}",
    ]

    if max_iterations > 0:
        extra_args.append(f"--num_iterations={max_iterations}")

    # Run training with torchrun
    run_torchrun_command("scripts.base_train", NUM_GPUS_BASE, extra_args)

    # Commit checkpoints to volume
    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("✓ Base model training completed!")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/base/")
    print("=" * 80)

    return {
        "status": "completed",
        "depth": depth,
        "device_batch_size": device_batch_size,
        "num_gpus": NUM_GPUS_BASE,
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/base",
    }

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_EVAL}",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def evaluate_base_model(max_per_task: int = 500):
    """
    Evaluate base model on CORE benchmark.

    CORE is a comprehensive evaluation of language model capabilities.

    Args:
        max_per_task: Max examples per task (-1 for all)
    """
    print("=" * 80)
    print("EVALUATING BASE MODEL - CORE METRIC")
    print("=" * 80)

    extra_args = []
    if max_per_task > 0:
        extra_args.append(f"--core_metric_max_per_task={max_per_task}")

    run_torchrun_command("scripts.base_eval", NUM_GPUS_EVAL, extra_args)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("✓ Base model evaluation completed!")
    print("=" * 80)

    return {"status": "completed"}

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_EVAL}",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def evaluate_base_loss():
    """
    Evaluate base model validation loss (bits per byte).

    This measures how well the model compresses the validation data.
    """
    print("=" * 80)
    print("EVALUATING BASE MODEL - VALIDATION LOSS")
    print("=" * 80)

    run_torchrun_command("scripts.base_loss", NUM_GPUS_EVAL)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("✓ Base loss evaluation completed!")
    print("=" * 80)

    return {"status": "completed"}

# =============================================================================
# STAGE 4: MIDTRAINING
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_MID}",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def train_mid_model(
    device_batch_size: int = 32,
    wandb_run: str = "dummy",
):
    """
    Midtrain the model on conversation data.

    Teaches the model:
    - Conversation special tokens (<|user_start|>, <|assistant_end|>, etc.)
    - Tool use (calculator)
    - Multiple choice question format

    Training data:
    - SmolTalk: 460K general conversations
    - MMLU auxiliary: 100K multiple choice problems
    - GSM8K: 8K math problems with calculator tool

    Duration: ~30-45 minutes on 8 GPUs

    Args:
        device_batch_size: Batch size per GPU
        wandb_run: WandB run name
    """
    print("=" * 80)
    print("MIDTRAINING - TEACHING CONVERSATION TOKENS")
    print("=" * 80)
    print(f"Device batch size: {device_batch_size}")
    print(f"Number of GPUs: {NUM_GPUS_MID}")
    print()

    extra_args = [
        f"--device_batch_size={device_batch_size}",
        f"--run={wandb_run}",
    ]

    run_torchrun_command("scripts.mid_train", NUM_GPUS_MID, extra_args)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("✓ Midtraining completed!")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/mid/")
    print("=" * 80)

    return {
        "status": "completed",
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/mid",
    }

# =============================================================================
# STAGE 5: SUPERVISED FINE-TUNING (SFT)
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_SFT}",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def train_sft_model(
    device_batch_size: int = 4,  # Smaller batch size for SFT
    num_epochs: int = 1,
    wandb_run: str = "dummy",
    source: str = "mid",  # "base" or "mid"
):
    """
    Supervised fine-tuning on task-specific data.

    Fine-tunes the model on a mixture of tasks:
    - MMLU: General knowledge
    - ARC: Science reasoning
    - GSM8K: Math with tool use
    - HumanEval: Code generation
    - SmolTalk: Conversations

    Duration: ~30-45 minutes on 8 GPUs

    Args:
        device_batch_size: Batch size per GPU (smaller for SFT)
        num_epochs: Number of training epochs
        wandb_run: WandB run name
        source: Base model to start from ("base" or "mid")
    """
    print("=" * 80)
    print("SUPERVISED FINE-TUNING")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Device batch size: {device_batch_size}")
    print(f"Number of GPUs: {NUM_GPUS_SFT}")
    print(f"Epochs: {num_epochs}")
    print()

    extra_args = [
        f"--device_batch_size={device_batch_size}",
        f"--num_epochs={num_epochs}",
        f"--run={wandb_run}",
        f"--source={source}",
    ]

    run_torchrun_command("scripts.chat_sft", NUM_GPUS_SFT, extra_args)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("✓ SFT completed!")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/sft/")
    print("=" * 80)

    return {
        "status": "completed",
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/sft",
    }

# =============================================================================
# STAGE 6: REINFORCEMENT LEARNING (OPTIONAL)
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_RL}",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def train_rl_model(
    device_batch_size: int = 8,
    num_epochs: int = 1,
    wandb_run: str = "dummy",
    source: str = "sft",
):
    """
    Reinforcement learning on GSM8K (optional).

    Uses simplified GRPO/REINFORCE to improve math reasoning.
    Only trains on GSM8K problems.

    Duration: ~30-45 minutes on 8 GPUs

    Args:
        device_batch_size: Batch size per GPU
        num_epochs: Number of training epochs
        wandb_run: WandB run name
        source: Model to start from (usually "sft")
    """
    print("=" * 80)
    print("REINFORCEMENT LEARNING ON GSM8K")
    print("=" * 80)
    print(f"Source: {source}")
    print(f"Device batch size: {device_batch_size}")
    print(f"Number of GPUs: {NUM_GPUS_RL}")
    print(f"Epochs: {num_epochs}")
    print()

    extra_args = [
        f"--device_batch_size={device_batch_size}",
        f"--num_epochs={num_epochs}",
        f"--run={wandb_run}",
        f"--source={source}",
    ]

    run_torchrun_command("scripts.chat_rl", NUM_GPUS_RL, extra_args)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("✓ RL training completed!")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/rl/")
    print("=" * 80)

    return {
        "status": "completed",
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/rl",
    }

# =============================================================================
# STAGE 7: EVALUATION
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_EVAL}",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def evaluate_chat_model(
    source: str = "sft",  # "mid", "sft", or "rl"
    tasks: str = "all",   # "all" or comma-separated: "ARC-Easy,GSM8K,HumanEval"
):
    """
    Evaluate the chat model on benchmark tasks.

    Available tasks:
    - ARC-Easy: Elementary science questions
    - ARC-Challenge: More difficult science questions
    - GSM8K: Grade school math problems
    - HumanEval: Code generation
    - MMLU: Massive multitask language understanding
    - ChatCORE: Conversation evaluation

    Args:
        source: Which checkpoint to evaluate ("mid", "sft", or "rl")
        tasks: Tasks to run ("all" or comma-separated list)
    """
    print("=" * 80)
    print(f"EVALUATING CHAT MODEL - {source.upper()}")
    print("=" * 80)
    print(f"Tasks: {tasks}")
    print()

    extra_args = ["-i", source]

    if tasks != "all":
        extra_args.extend(["-a", tasks])

    run_torchrun_command("scripts.chat_eval", NUM_GPUS_EVAL, extra_args)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print(f"✓ Evaluation of {source} model completed!")
    print("=" * 80)

    return {
        "status": "completed",
        "source": source,
        "tasks": tasks,
    }

# =============================================================================
# STAGE 8: INFERENCE
# =============================================================================

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_INFERENCE}",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def chat_cli(
    source: str = "sft",
    prompt: str = "",
    temperature: float = 0.6,
    top_k: int = 50,
):
    """
    Chat with the model via command line interface.

    Args:
        source: Which checkpoint to use ("mid", "sft", or "rl")
        prompt: Prompt for the model (empty for interactive mode)
        temperature: Sampling temperature (0.1=boring, 1.5=wild)
        top_k: Top-k sampling parameter
    """
    import subprocess

    print("=" * 80)
    print(f"CHAT CLI - {source.upper()} MODEL")
    print("=" * 80)

    cmd = [
        "python", "-m", "scripts.chat_cli",
        "-i", source,
        "-t", str(temperature),
        "-k", str(top_k),
    ]

    if prompt:
        cmd.extend(["-p", prompt])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Chat CLI failed with code {result.returncode}")

    return {
        "status": "completed",
        "source": source,
        "prompt": prompt,
    }

@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_INFERENCE}",
    volumes=VOLUME_CONFIG,
    timeout=4 * HOURS,  # Long timeout for web server
    concurrency_limit=100,  # Allow up to 100 concurrent requests
)
def chat_web(
    source: str = "sft",
    port: int = 8000,
    temperature: float = 0.8,
    top_k: int = 50,
    max_tokens: int = 512,
):
    """
    Serve the chat model via a web UI (like ChatGPT).

    This starts a FastAPI server with a beautiful web interface.

    Args:
        source: Which checkpoint to use ("mid", "sft", or "rl")
        port: Port to serve on (default: 8000)
        temperature: Default sampling temperature
        top_k: Default top-k sampling
        max_tokens: Default max tokens per response
    """
    import subprocess

    print("=" * 80)
    print(f"STARTING WEB UI - {source.upper()} MODEL")
    print("=" * 80)
    print(f"Port: {port}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
    print(f"Max tokens: {max_tokens}")
    print()

    cmd = [
        "python", "-m", "scripts.chat_web",
        "-i", source,
        "-p", str(port),
        "-t", str(temperature),
        "-k", str(top_k),
        "-m", str(max_tokens),
        "--host", "0.0.0.0",
    ]

    print(f"Running: {' '.join(cmd)}")
    print("\n" + "=" * 80)
    print(f"Web UI will be available at: http://localhost:{port}")
    print("=" * 80)
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Web server failed with code {result.returncode}")

    return {
        "status": "completed",
        "source": source,
        "port": port,
    }

# =============================================================================
# MAIN PIPELINE ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(
    # Pipeline stages to run
    run_download: bool = True,
    run_tokenizer: bool = True,
    run_base: bool = True,
    run_mid: bool = True,
    run_sft: bool = True,
    run_rl: bool = False,      # Optional by default
    run_eval: bool = True,
    run_inference: bool = True,

    # Configuration
    num_data_shards: int = 240,  # 240 for speedrun, 8 for quick test
    depth: int = 20,             # 20=561M params, 26=1B params, 12=minimal test
    device_batch_size_base: int = 32,
    device_batch_size_sft: int = 4,
    wandb_run: str = "dummy",
):
    """
    Run the complete nanochat pipeline from scratch.

    This orchestrates all stages in sequence:
    1. Download FineWeb dataset
    2. Train + evaluate tokenizer
    3. Train + evaluate base model
    4. Train + evaluate mid model
    5. Train + evaluate SFT model
    6. (Optional) Train + evaluate RL model
    7. Run final inference test

    Modes:
    - Full Speedrun (4 hours, $96): num_data_shards=240, depth=20
    - Quick Test (1 hour, $24): num_data_shards=8, depth=12
    - GPT-2 Grade (12 hours, $288): num_data_shards=450, depth=26

    Args:
        run_*: Flags to enable/disable pipeline stages
        num_data_shards: Dataset size (8=test, 240=speedrun, 450=GPT-2)
        depth: Model size (12=tiny, 20=speedrun, 26=GPT-2)
        device_batch_size_*: Batch sizes (reduce if OOM)
        wandb_run: WandB run name ("dummy" to disable logging)
    """
    print("=" * 80)
    print("🚀 NANOCHAT TRAINING PIPELINE")
    print("=" * 80)
    print(f"Mode: {'Speedrun' if num_data_shards >= 240 else 'Quick Test'}")
    print(f"Data shards: {num_data_shards}")
    print(f"Model depth: {depth}")
    print(f"WandB run: {wandb_run}")
    print("=" * 80)
    print()

    # Stage 1: Download dataset
    if run_download:
        print("📥 Stage 1/8: Downloading dataset...")
        download_dataset.remote(num_shards=num_data_shards)

    # Stage 2: Tokenizer
    if run_tokenizer:
        print("\n🔤 Stage 2/8: Training tokenizer...")
        train_tokenizer.remote()
        print("Evaluating tokenizer...")
        evaluate_tokenizer.remote()

    # Stage 3: Base model
    if run_base:
        print("\n🏋️ Stage 3/8: Training base model...")
        train_base_model.remote(
            depth=depth,
            device_batch_size=device_batch_size_base,
            wandb_run=wandb_run
        )
        if run_eval:
            print("Evaluating base model (CORE)...")
            evaluate_base_model.remote()
            print("Evaluating base model (loss)...")
            evaluate_base_loss.remote()

    # Stage 4: Midtraining
    if run_mid:
        print("\n💬 Stage 4/8: Midtraining (conversation tokens)...")
        train_mid_model.remote(
            device_batch_size=device_batch_size_base,
            wandb_run=wandb_run
        )
        if run_eval:
            print("Evaluating mid model...")
            evaluate_chat_model.remote(source="mid")

    # Stage 5: SFT
    if run_sft:
        print("\n🎯 Stage 5/8: Supervised fine-tuning...")
        train_sft_model.remote(
            device_batch_size=device_batch_size_sft,
            wandb_run=wandb_run,
            source="mid"
        )
        if run_eval:
            print("Evaluating SFT model...")
            evaluate_chat_model.remote(source="sft")

    # Stage 6: RL (optional)
    if run_rl:
        print("\n🎮 Stage 6/8: Reinforcement learning...")
        train_rl_model.remote(wandb_run=wandb_run)
        if run_eval:
            print("Evaluating RL model...")
            evaluate_chat_model.remote(source="rl", tasks="GSM8K")

    # Stage 7: Final inference test
    if run_inference:
        print("\n✨ Stage 7/8: Testing inference...")
        final_source = "rl" if run_rl else "sft"
        chat_cli.remote(
            source=final_source,
            prompt="Why is the sky blue?"
        )

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Chat via CLI: modal run TrainNanochatModal.py::chat_cli --source=sft")
    print("2. Launch Web UI: modal run TrainNanochatModal.py::chat_web --source=sft")
    print("3. Run more evals: modal run TrainNanochatModal.py::evaluate_chat_model --source=sft")
    print()
