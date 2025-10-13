# Training Nanochat on Modal: Build Your Own ChatGPT from Scratch

📄 **[View Complete Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/TrainNanochatModal.py)**

So you've fine-tuned models with LoRA, trained small GPTs, even tackled multi-GPU training. Now let's go full circle and build ChatGPT from absolute scratch - tokenizer training, base model pretraining, conversation fine-tuning, the works.

We're talking about the complete Andrej Karpathy nanochat speedrun. Everything from "raw text on the internet" to "functioning ChatGPT clone that can have conversations and use tools."

## Why Nanochat?

Here's the thing - if you want to truly understand how ChatGPT works, you can't just fine-tune a pre-trained model. You need to see the entire pipeline.

Andrej Karpathy built nanochat as the most comprehensive educational implementation of modern LLM training. It's not a toy - it's the real deal, just scaled down to be understandable and trainable on reasonable hardware.

**What makes nanochat special:**
- **Complete pipeline** - Every single step from tokenizer to deployment
- **Production techniques** - Same methods used by OpenAI, Anthropic, etc.
- **Educational focus** - Clean, readable code with excellent documentation
- **Proven results** - Actually produces working chat models with tool use
- **Reasonable scale** - Train on 4-8 GPUs instead of thousands

This is basically "here's how we built ChatGPT" but in a form you can actually run and understand. The original repo assumes you have a local GPU cluster. We're taking that exact pipeline and making it run on Modal's serverless infrastructure.

## What We're Building

This isn't just training a model. We're building the entire stack:

**Stage 1: Download Dataset** - Grab FineWeb-edu (100B tokens of high-quality text)
**Stage 2: Train Tokenizer** - Build a custom BPE tokenizer (like GPT-4 uses)
**Stage 3: Base Pretraining** - Train a GPT on raw internet text
**Stage 4: Midtraining** - Teach conversation format and tool use
**Stage 5: Supervised Fine-tuning** - Train on specific tasks (code, math, chat)
**Stage 6: Reinforcement Learning** - Optional GRPO on math problems
**Stage 7: Evaluation** - Measure performance on real benchmarks
**Stage 8: Inference** - Chat with your model (CLI and web UI)

The beauty of this pipeline? Each stage is independent. Screw up fine-tuning? Just re-run that step. Want to try different hyperparameters? Preprocessing is already done.

Here's what the complete flow looks like:

```
┌─────────────────────┐
│  Download FineWeb   │  (CPU - cheap, run once)
│  100B tokens        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Train Tokenizer    │  (1 GPU - 30-60 min)
│  Custom BPE 65K     │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Pretrain Base      │  (4-8 GPUs - 2-4 hours)
│  Language Model     │  ← The big one
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Midtrain           │  (4 GPUs - 30-45 min)
│  Conversation       │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  SFT                │  (4 GPUs - 30-45 min)
│  Task-specific      │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  RL (Optional)      │  (4 GPUs - 30-45 min)
│  Math reasoning     │
└──────────┬──────────┘
           │
      ┌────┴────┐
      │         │
┌─────▼───┐ ┌──▼──────┐
│  Chat   │ │  Eval   │
│  CLI/Web│ │  Metrics│
└─────────┘ └─────────┘
```

**Total time for full speedrun:** ~4-5 hours on 8× A100-80GB
**Total cost:** ~$100-150 for the complete pipeline

Compare this to OpenAI spending millions... yeah, we're doing pretty well.

## Getting Started

### Install Modal

You know the drill:

```bash
pip install modal
```

### Authenticate

```bash
modal setup
```

Or with API keys:

```bash
export MODAL_TOKEN_ID=<your_token_id>
export MODAL_TOKEN_SECRET=<your_token_secret>
```

### Set Up Your Secrets

For this pipeline, you'll want:
- **Hugging Face token** - For downloading datasets
- **Weights & Biases API key** - For tracking training (highly recommended!)

**Create the Modal secret:**

```bash
modal secret create nanochat-secrets \
  WANDB_API_KEY=xxxxxxxxxxxxx \
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
```

**Get your tokens:**
- HF token: [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
- W&B key: [wandb.ai/authorize](https://wandb.ai/authorize)

**Or use a .env file (local development):**

```bash
WANDB_API_KEY=your_key
HUGGINGFACE_TOKEN=hf_your_token
```

The script tries `.env` first, then falls back to Modal secrets.

### Clone Nanochat

This is important - we need the nanochat repo locally:

```bash
cd /path/to/ServerLessFinetuning
git clone https://github.com/karpathy/nanochat.git
```

Your folder structure should be:

```
ServerLessFinetuning/
├── TrainNanochatModal.py    # Your Modal script
└── nanochat/                 # Cloned repo
    ├── scripts/
    ├── nanochat/
    └── rustbpe/
```

**Why clone it?** Modal copies this entire directory into the container image. This way we have all of nanochat's scripts and utilities available.

## Understanding the Pipeline

Let me walk you through what makes this different from the other tutorials. This isn't fine-tuning - this is building everything from scratch.

### The Full Speedrun

Nanochat is designed as a "speedrun" - train a working ChatGPT in one go. Here's what each stage does:

**Stage 1: Dataset Download**
- Downloads FineWeb-edu-100B from HuggingFace
- 240 shards × 250M characters = ~60B characters
- Enough to train a 561M parameter model (Chinchilla optimal)

**Stage 2: Tokenizer Training**
- Trains a BPE tokenizer on 2B characters
- Creates a 65K vocab (2^16 tokens)
- Same approach as GPT-4

**Stage 3: Base Pretraining**
- Trains a GPT from random initialization
- Uses Muon optimizer (better than Adam for LLMs)
- Learns language from raw text

**Stage 4: Midtraining**
- Teaches conversation format (user/assistant turns)
- Adds tool use (calculator)
- Trains on SmolTalk + MMLU + GSM8K

**Stage 5: Supervised Fine-tuning**
- Task-specific training
- MMLU (knowledge), ARC (reasoning), GSM8K (math), HumanEval (code)

**Stage 6: Reinforcement Learning (Optional)**
- GRPO/REINFORCE on math problems
- Improves reasoning through self-play

**Stage 7: Evaluation**
- CORE metric (comprehensive benchmark)
- Task-specific evals (ARC, GSM8K, HumanEval, MMLU)

**Stage 8: Inference**
- Chat CLI for interactive testing
- FastAPI web UI for demos

## Configuration and Setup

Alright, let's dive into the code. The configuration is straightforward but important.

### App and Volumes

```python
from modal import App, Image as ModalImage, Volume, Secret

app = App("nanochat-training")

# Two volumes: one for data, one for checkpoints
data_volume = Volume.from_name("nanochat-data", create_if_missing=True)
checkpoint_volume = Volume.from_name("nanochat-checkpoints", create_if_missing=True)

VOLUME_CONFIG = {
    "/data": data_volume,          # Dataset and cache
    "/checkpoints": checkpoint_volume,  # Model checkpoints
}
```

**Why two volumes?**
- Data volume: FineWeb shards, tokenizer, eval data (~30GB)
- Checkpoint volume: Model weights, training state (~20GB)
- Separation makes it easier to manage and debug

### Configuration Constants

```python
MINUTES = 60
HOURS = 60 * 60

# GPU type - can be changed based on your needs
GPU_TYPE = "a100-80gb"

# Multi-GPU configuration (nanochat supports 1-8 GPUs)
NUM_GPUS_BASE = 4        # Base pretraining
NUM_GPUS_MID = 4         # Midtraining
NUM_GPUS_SFT = 4         # Supervised fine-tuning
NUM_GPUS_RL = 4          # Reinforcement learning
NUM_GPUS_EVAL = 4        # Evaluation
NUM_GPUS_TOKENIZER = 1   # Tokenizer (single GPU)
NUM_GPUS_INFERENCE = 1   # Inference (single GPU)

BASE_DIR = "/data/.cache/nanochat"  # Everything goes here
```

**GPU scaling:**
- 1 GPU: Full pipeline takes ~24 hours (~$84)
- 4 GPUs: Full pipeline takes ~6 hours (~$96)
- 8 GPUs: Full pipeline takes ~4 hours (~$112)

The sweet spot is 4 GPUs - good parallelism without too much communication overhead.

### Secrets Setup

```python
# Try .env first, then Modal secrets
try:
    nanochat_secret = Secret.from_dotenv()
    print("Loaded secrets from .env file")
except Exception:
    try:
        nanochat_secret = Secret.from_name("nanochat-secrets")
        print("Loaded secrets from Modal")
    except Exception:
        nanochat_secret = None
        print("No secrets found - WandB logging disabled")
```

This is graceful - works with or without secrets. If you don't have W&B, training still works, you just don't get the nice dashboards.

## Building the Container Image

This is the most complex image we've built yet. We need CUDA, PyTorch, Rust (for the tokenizer), and all of nanochat's dependencies.

### Why This Image is Different

```python
NANOCHAT_IMAGE = (
    # NVIDIA CUDA 12.8 with Python 3.11
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")

    # System dependencies
    .apt_install("git", "build-essential", "curl", "wget", "unzip")

    # Install Rust (needed for tokenizer)
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> $HOME/.bashrc",
    )

    # Install uv (fast Python package installer)
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> $HOME/.bashrc",
    )

    # Copy nanochat repo into the image
    .add_local_dir(local_path="nanochat", remote_path="/root/nanochat", copy=True)

    # Set working directory
    .workdir("/root/nanochat")

    # Install Python dependencies AND build Rust tokenizer
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && uv sync && uv run maturin develop --release --manifest-path rustbpe/Cargo.toml'"
    )

    # Environment variables
    .env({
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": "/data/.cache/nanochat",
        "HF_HOME": "/data/.cache/huggingface",
    })
)
```

**Key points:**

1. **Rust installation:** Nanochat's tokenizer is written in Rust for speed. We need the full Rust toolchain.

2. **uv sync:** This reads `pyproject.toml` and installs all dependencies in a virtual environment. Much faster than pip.

3. **maturin develop:** Builds the Rust tokenizer and makes it importable from Python. This is the magic that makes nanochat's tokenizer so fast.

4. **add_local_dir:** Copies your local nanochat clone into the image. This is why you need to clone it first.

> **⏰ Build time warning:** First build takes 15-20 minutes. The Rust tokenizer compilation is the slow part. But Modal caches everything - subsequent runs are instant.

## Helper Functions

Before we get to the stages, let's look at the helper functions that make everything work.

### Setup Functions

```python
def setup_base_dir():
    """Create directory structure for nanochat artifacts."""
    import os

    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(f"{BASE_DIR}/base_data", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/tokenizer", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/checkpoints", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/eval_bundle", exist_ok=True)
    os.makedirs(f"{BASE_DIR}/report", exist_ok=True)

def setup_secrets():
    """Set up environment variables from secrets."""
    import os

    if "WANDB_API_KEY" in os.environ:
        print("WandB API key found")
    else:
        print("WandB API key not found - logging disabled")

    if "HUGGINGFACE_TOKEN" in os.environ:
        os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
        print("HuggingFace token found")
    else:
        print("HuggingFace token not found")
```

Simple but important - creates all the directories nanochat expects and sets up authentication.

### Torchrun Helper

```python
def run_torchrun_command(script: str, num_gpus: int, extra_args: list = None):
    """Run nanochat script with torchrun for multi-GPU training."""
    import subprocess

    if extra_args is None:
        extra_args = []

    extra_args_str = " ".join(extra_args) if extra_args else ""
    cmd = f"cd /root/nanochat && uv run torchrun --standalone --nproc_per_node={num_gpus} -m {script}"

    if extra_args:
        cmd += f" -- {extra_args_str}"

    print(f"Running: {cmd}")
    result = subprocess.run(["bash", "-c", cmd], capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}")

    return result
```

**What torchrun does:**
- Spawns one process per GPU
- Sets up distributed training (NCCL backend)
- Handles rank assignment and synchronization
- Makes multi-GPU training "just work"

This is how nanochat does distributed training without writing custom distributed code.

## Stage 1: Dataset Download

Let's start with the data. We're downloading FineWeb-edu, which is basically "the good parts of the internet."

```python
@app.function(
    image=NANOCHAT_IMAGE,
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
    # No GPU - saves money!
)
def download_dataset(num_shards: int = 240):
    """
    Download FineWeb dataset shards from HuggingFace.

    Each shard is ~250M characters (~100MB compressed).
    - Full speedrun: 240 shards (~60B characters, ~24GB)
    - Testing: 8 shards (~2B characters, ~800MB)
    """
    import subprocess

    setup_base_dir()

    print("=" * 80)
    print(f"DOWNLOADING FINEWEB DATASET - {num_shards} SHARDS")
    print("=" * 80)
    print(f"Total data: ~{num_shards * 250 / 1000:.1f}B characters (~{num_shards * 100 / 1024:.1f}GB)")
    print()

    result = subprocess.run(
        ["bash", "-c", f"cd /root/nanochat && uv run python -m nanochat.dataset -n {num_shards}"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Dataset download failed with code {result.returncode}")

    data_volume.commit()

    print("\n" + "=" * 80)
    print(f"Downloaded {num_shards} shards successfully")
    print("=" * 80)

    return {
        "status": "completed",
        "num_shards": num_shards,
        "data_dir": f"{BASE_DIR}/base_data",
    }
```

**How many shards do you need?**

| Model Size | Parameters | Chinchilla Tokens | Characters Needed | Shards |
|------------|-----------|-------------------|-------------------|--------|
| Tiny test | ~100M | 2B | ~10B | 40 |
| Quick test | ~200M | 4B | ~20B | 80 |
| Speedrun | 561M (d20) | 11B | ~54B | 240 |
| GPT-2 grade | ~1B (d26) | 20B | ~96B | 400 |

For testing, use 8 shards. For the real speedrun, use 240.

**Run it:**

```bash
# Full speedrun dataset
modal run TrainNanochatModal.py::download_dataset

# Quick test dataset
modal run TrainNanochatModal.py::download_dataset --num-shards=8
```

First download takes 30-60 minutes depending on internet speed. Cached after that.

## Stage 2: Tokenizer Training

Now we train a custom tokenizer on the FineWeb data. This is similar to how GPT-4's tokenizer was trained.

```python
@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_TOKENIZER}",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def train_tokenizer(
    max_chars: int = 2_000_000_000,  # 2 billion characters
    vocab_size: int = 65536,          # 65K vocab (2^16)
    doc_cap: int = 10000,              # Max chars per document
):
    """
    Train a custom BPE tokenizer on FineWeb data.
    Training takes 30-60 minutes on a single GPU.
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

    cmd = f"cd /root/nanochat && uv run python -m scripts.tok_train --max_chars={max_chars} --vocab_size={vocab_size} --doc_cap={doc_cap}"

    print(f"Running: {cmd}")
    result = subprocess.run(["bash", "-c", cmd], capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer training failed with code {result.returncode}")

    data_volume.commit()
    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("Tokenizer training completed")
    print(f"Tokenizer saved to {BASE_DIR}/tokenizer/")
    print("=" * 80)

    return {
        "status": "completed",
        "max_chars": max_chars,
        "vocab_size": vocab_size,
        "tokenizer_dir": f"{BASE_DIR}/tokenizer",
    }
```

**Why train a custom tokenizer?**

You might wonder "why not use GPT-2's tokenizer?" Here's why:

1. **Domain-specific:** FineWeb has different token distributions than GPT-2's training data
2. **Efficiency:** Custom tokenizers compress better (lower bits per byte)
3. **Control:** You decide vocab size, special tokens, etc.
4. **Learning:** Understanding tokenization is crucial for understanding LLMs

**The training process:**
- Samples 2B characters from FineWeb
- Runs Byte Pair Encoding (BPE) to find common subwords
- Creates a 65K vocabulary
- Saves tokenizer model and merges

**Run it:**

```bash
modal run TrainNanochatModal.py::train_tokenizer
```

Takes 30-60 minutes on a single GPU. You can also evaluate it:

```bash
modal run TrainNanochatModal.py::evaluate_tokenizer
```

This shows you the compression ratio and other metrics.

## Stage 3: Base Model Pretraining

Here's the big one. We're training a GPT from scratch on internet text. This is where most of the compute goes.

### Model Architecture

Nanochat uses a simple but effective architecture:

```python
depth: int = 20  # Model depth parameter

# Derived dimensions:
model_dim = depth * 64      # 20 * 64 = 1280
num_heads = model_dim / 128 # 1280 / 128 = 10
num_layers = depth          # 20 layers

# Total parameters: ~561M
```

**Model sizes:**
- depth=12: ~200M params (quick test)
- depth=20: ~561M params (default speedrun)
- depth=26: ~1B params (GPT-2 grade)

### The Training Function

```python
@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_BASE}",
    volumes=VOLUME_CONFIG,
    secrets=[nanochat_secret] if nanochat_secret else [],
    timeout=8 * HOURS,
)
def train_base_model(
    depth: int = 20,
    device_batch_size: int = 32,
    max_iterations: int = -1,  # -1 = auto from Chinchilla
    wandb_run: str = "dummy",
):
    """
    Pretrain the base GPT model on FineWeb.

    Model sizes: depth=20 (561M params), depth=26 (1B params)
    Training duration: ~2-3 hours on 8 GPUs, ~16-24 hours on 1 GPU
    """
    import subprocess
    import os

    setup_base_dir()
    setup_secrets()

    # Download eval bundle if needed (for CORE metric)
    eval_bundle_path = f"{BASE_DIR}/eval_bundle"
    if not os.path.exists(eval_bundle_path):
        print("Downloading eval bundle...")
        subprocess.run([
            "curl", "-L", "-o", "eval_bundle.zip",
            "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip",
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

    extra_args = [
        f"--depth={depth}",
        f"--device_batch_size={device_batch_size}",
        f"--run={wandb_run}",
    ]

    if max_iterations > 0:
        extra_args.append(f"--num_iterations={max_iterations}")

    run_torchrun_command("scripts.base_train", NUM_GPUS_BASE, extra_args)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("Base model training completed")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/base/")
    print("=" * 80)

    return {
        "status": "completed",
        "depth": depth,
        "device_batch_size": device_batch_size,
        "num_gpus": NUM_GPUS_BASE,
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/base",
    }
```

**What happens during training:**

1. **Initialization:** Random weights, Muon optimizer setup
2. **Training loop:** Read batches, forward pass, backward pass, update weights
3. **Evaluation:** Every N steps, compute validation loss and CORE metric
4. **Checkpointing:** Save model state every N steps
5. **Logging:** Send metrics to W&B (if configured)

**The Muon Optimizer:**

Nanochat uses Muon for linear layers and AdamW for embeddings. Muon is a new optimizer that:
- Converges faster than Adam for transformers
- Uses less memory (no momentum for linear layers)
- More stable for large learning rates

It's basically "what if we applied optimizer improvements from recent research?"

### Training Duration and Cost

| Setup | GPUs | Time | Cost |
|-------|------|------|------|
| Quick test (d12, 8 shards) | 1 | ~2 hours | ~$7 |
| Quick test (d12, 8 shards) | 4 | ~30 min | ~$7 |
| Full speedrun (d20, 240 shards) | 1 | ~20 hours | ~$70 |
| Full speedrun (d20, 240 shards) | 4 | ~5 hours | ~$70 |
| Full speedrun (d20, 240 shards) | 8 | ~3 hours | ~$84 |
| GPT-2 grade (d26, 400 shards) | 8 | ~10 hours | ~$280 |

**Run it:**

```bash
# Quick test (make sure it works)
modal run TrainNanochatModal.py::train_base_model \
  --depth=12 \
  --device-batch-size=32 \
  --max-iterations=1000

# Full speedrun
modal run TrainNanochatModal.py::train_base_model \
  --depth=20 \
  --device-batch-size=32 \
  --wandb-run="my-speedrun-$(date +%Y%m%d)"
```

**Monitor training:**
- Modal dashboard shows real-time logs and GPU utilization
- W&B shows loss curves, learning rate schedule, CORE metric
- All 4 (or 8) GPUs should be at ~95-100% utilization

### Evaluation

After training (or during), you can evaluate:

```bash
# CORE metric (comprehensive benchmark)
modal run TrainNanochatModal.py::evaluate_base_model

# Validation loss (bits per byte)
modal run TrainNanochatModal.py::evaluate_base_loss
```

CORE metric measures general language understanding. A good d20 model gets ~0.35-0.40. For comparison:
- Random: 0.0
- Llama 3 8B: ~0.50
- GPT-4: ~0.60

## Stage 4: Midtraining

Now we teach the model how to have conversations. This is where it learns special tokens like `<|user_start|>`, `<|assistant_end|>`, etc.

```python
@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_MID}",
    volumes=VOLUME_CONFIG,
    secrets=[nanochat_secret] if nanochat_secret else [],
    timeout=2 * HOURS,
)
def train_mid_model(
    device_batch_size: int = 32,
    wandb_run: str = "dummy",
):
    """
    Midtrain the model on conversation data.

    Teaches conversation tokens, tool use, and multiple choice format.
    Duration: ~30-45 minutes on 8 GPUs
    """
    setup_secrets()

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
    print("Midtraining completed")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/mid/")
    print("=" * 80)

    return {
        "status": "completed",
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/mid",
    }
```

**What midtraining teaches:**

1. **Conversation format:**
   ```
   <|user_start|>What is 2+2?<|user_end|>
   <|assistant_start|>2+2 equals 4.<|assistant_end|>
   ```

2. **Tool use:**
   ```
   <|user_start|>What is 123 * 456?<|user_end|>
   <|assistant_start|><calculator>123*456</calculator>
   The answer is 56088.<|assistant_end|>
   ```

3. **Multiple choice:**
   ```
   Question: What is the capital of France?
   A) London  B) Paris  C) Berlin  D) Madrid
   Answer: B
   ```

**Training data:**
- SmolTalk: 460K conversations
- MMLU auxiliary: 100K multiple choice
- GSM8K: 8K math problems with calculator

**Run it:**

```bash
modal run TrainNanochatModal.py::train_mid_model
```

Takes 30-45 minutes on 4-8 GPUs, costs ~$5-7.

## Stage 5: Supervised Fine-tuning

Now we specialize the model on specific tasks: knowledge, reasoning, math, code, and chat.

```python
@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_SFT}",
    volumes=VOLUME_CONFIG,
    secrets=[nanochat_secret] if nanochat_secret else [],
    timeout=2 * HOURS,
)
def train_sft_model(
    device_batch_size: int = 4,
    num_epochs: int = 1,
    wandb_run: str = "dummy",
    source: str = "mid",
):
    """
    Supervised fine-tuning on task-specific data.

    Trains on MMLU, ARC, GSM8K, HumanEval, and SmolTalk.
    Duration: ~30-45 minutes on 8 GPUs
    """
    setup_secrets()

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
    print("SFT completed")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/sft/")
    print("=" * 80)

    return {
        "status": "completed",
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/sft",
    }
```

**Training mixture:**

| Dataset | Examples | Task | Weight |
|---------|----------|------|--------|
| MMLU | ~14K | General knowledge | 25% |
| ARC-Easy | ~2.4K | Science reasoning | 15% |
| ARC-Challenge | ~1.2K | Hard science | 15% |
| GSM8K | ~7.5K | Math with tools | 20% |
| HumanEval | ~164 | Code generation | 5% |
| SmolTalk | ~50K | Conversations | 20% |

The mixture is designed to produce a well-rounded model that can chat, reason, do math, and write code.

**Run it:**

```bash
# Start from midtrained model (recommended)
modal run TrainNanochatModal.py::train_sft_model --source=mid

# Or start from base model (skip midtraining)
modal run TrainNanochatModal.py::train_sft_model --source=base
```

Takes 30-45 minutes on 4-8 GPUs, costs ~$5-7.

## Stage 6: Reinforcement Learning (Optional)

This is optional but improves math reasoning significantly.

```python
@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_RL}",
    volumes=VOLUME_CONFIG,
    secrets=[nanochat_secret] if nanochat_secret else [],
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

    Uses GRPO/REINFORCE to improve math reasoning.
    Duration: ~30-45 minutes on 8 GPUs
    """
    setup_secrets()

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
    print("RL training completed")
    print(f"Checkpoints saved to {BASE_DIR}/checkpoints/rl/")
    print("=" * 80)

    return {
        "status": "completed",
        "checkpoint_dir": f"{BASE_DIR}/checkpoints/rl",
    }
```

**How RL works here:**

1. Model generates multiple answers to a math problem
2. Answers that get the right solution get positive reward
3. Answers that get the wrong solution get negative reward
4. Update model to generate more correct answers

This is simplified GRPO (Group Relative Policy Optimization). It's like PPO but simpler and works well for math.

**Expected improvement:**
- GSM8K accuracy: 60% → 75% after RL
- ARC/MMLU: Stays about the same (RL only trains on math)

**Run it:**

```bash
modal run TrainNanochatModal.py::train_rl_model
```

Takes 30-45 minutes on 4-8 GPUs, costs ~$5-7.

## Stage 7: Evaluation

Now let's measure how good our model actually is. Nanochat includes comprehensive evaluation on real benchmarks.

```python
@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_EVAL}",
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def evaluate_chat_model(
    source: str = "sft",
    tasks: str = "all",
):
    """
    Evaluate the chat model on benchmark tasks.

    Available tasks: ARC-Easy, ARC-Challenge, GSM8K, HumanEval, MMLU, ChatCORE
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
    print(f"Evaluation of {source} model completed")
    print("=" * 80)

    return {
        "status": "completed",
        "source": source,
        "tasks": tasks,
    }
```

**Available benchmarks:**

| Benchmark | What it measures | Good score |
|-----------|------------------|------------|
| ARC-Easy | Elementary science | >60% |
| ARC-Challenge | Hard science | >35% |
| GSM8K | Grade school math | >60% |
| HumanEval | Code generation | >20% |
| MMLU | General knowledge | >50% |
| ChatCORE | Conversation quality | >0.40 |

**Run evaluation:**

```bash
# Evaluate all tasks
modal run TrainNanochatModal.py::evaluate_chat_model --source=sft

# Evaluate specific tasks
modal run TrainNanochatModal.py::evaluate_chat_model \
  --source=rl \
  --tasks="GSM8K,ARC-Challenge"

# Compare mid vs sft vs rl
modal run TrainNanochatModal.py::evaluate_chat_model --source=mid
modal run TrainNanochatModal.py::evaluate_chat_model --source=sft
modal run TrainNanochatModal.py::evaluate_chat_model --source=rl
```

Takes 30-60 minutes depending on which tasks you run. Results are saved to the volume and logged to W&B.

## Stage 8: Inference

Finally, let's chat with our model! Nanochat provides both a CLI and a web UI.

### Chat CLI

```python
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
    """Chat with the model via command line interface."""
    import subprocess

    print("=" * 80)
    print(f"CHAT CLI - {source.upper()} MODEL")
    print("=" * 80)

    cmd = f"cd /root/nanochat && uv run python -m scripts.chat_cli -i {source} -t {temperature} -k {top_k}"

    if prompt:
        escaped_prompt = prompt.replace('"', '\\"')
        cmd += f' -p "{escaped_prompt}"'

    print(f"Running: {cmd}")
    result = subprocess.run(["bash", "-c", cmd], capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Chat CLI failed with code {result.returncode}")

    return {
        "status": "completed",
        "source": source,
        "prompt": prompt,
    }
```

**Use it:**

```bash
# Single question
modal run TrainNanochatModal.py::chat_cli \
  --source=sft \
  --prompt="Why is the sky blue?"

# Interactive mode (empty prompt)
modal run TrainNanochatModal.py::chat_cli --source=sft

# With different sampling
modal run TrainNanochatModal.py::chat_cli \
  --source=rl \
  --temperature=0.8 \
  --top-k=100
```

### Chat Web UI

For a more user-friendly interface:

```python
@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_INFERENCE}",
    volumes=VOLUME_CONFIG,
    timeout=4 * HOURS,
    max_containers=2,
)
def chat_web(
    source: str = "sft",
    port: int = 8000,
    temperature: float = 0.8,
    top_k: int = 50,
    max_tokens: int = 512,
):
    """Serve the chat model via a web UI."""
    import subprocess

    print("=" * 80)
    print(f"STARTING WEB UI - {source.upper()} MODEL")
    print("=" * 80)
    print(f"Port: {port}")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
    print(f"Max tokens: {max_tokens}")
    print()

    cmd = f"cd /root/nanochat && uv run python -m scripts.chat_web -i {source} -p {port} -t {temperature} -k {top_k} -m {max_tokens} --host 0.0.0.0"

    print(f"Running: {cmd}")
    print("\n" + "=" * 80)
    print(f"Web UI will be available at: http://localhost:{port}")
    print("=" * 80)
    print()

    result = subprocess.run(["bash", "-c", cmd], capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Web server failed with code {result.returncode}")

    return {
        "status": "completed",
        "source": source,
        "port": port,
    }
```

**Deploy it:**

```bash
modal deploy TrainNanochatModal.py
```

Modal gives you a URL. Open it in your browser and you have a ChatGPT-like interface!

## Running the Complete Pipeline

Alright, let's put it all together. Here's how to run the full speedrun from scratch.

### The Main Pipeline

```python
@app.local_entrypoint()
def main(
    run_download: bool = True,
    run_tokenizer: bool = True,
    run_base: bool = True,
    run_mid: bool = True,
    run_sft: bool = True,
    run_rl: bool = False,
    run_eval: bool = True,
    run_inference: bool = True,
    num_data_shards: int = 240,
    depth: int = 20,
    device_batch_size_base: int = 32,
    device_batch_size_sft: int = 4,
    wandb_run: str = "dummy",
):
    """
    Run the complete nanochat pipeline from scratch.

    Configuration modes:
    - Full Speedrun (4h, $96): num_data_shards=240, depth=20
    - Quick Test (1h, $24): num_data_shards=8, depth=12
    - GPT-2 Grade (12h, $288): num_data_shards=450, depth=26
    """
    print("=" * 80)
    print("NANOCHAT TRAINING PIPELINE")
    print("=" * 80)
    print(f"Mode: {'Speedrun' if num_data_shards >= 240 else 'Quick Test'}")
    print(f"Data shards: {num_data_shards}")
    print(f"Model depth: {depth}")
    print(f"WandB run: {wandb_run}")
    print("=" * 80)
    print()

    if run_download:
        print("Stage 1/8: Downloading dataset...")
        download_dataset.remote(num_shards=num_data_shards)

    if run_tokenizer:
        print("\nStage 2/8: Training tokenizer...")
        train_tokenizer.remote()
        print("Evaluating tokenizer...")
        evaluate_tokenizer.remote()

    if run_base:
        print("\nStage 3/8: Training base model...")
        train_base_model.remote(
            depth=depth, device_batch_size=device_batch_size_base, wandb_run=wandb_run
        )
        if run_eval:
            print("Evaluating base model (CORE)...")
            evaluate_base_model.remote()
            print("Evaluating base model (loss)...")
            evaluate_base_loss.remote()

    if run_mid:
        print("\nStage 4/8: Midtraining (conversation tokens)...")
        train_mid_model.remote(
            device_batch_size=device_batch_size_base, wandb_run=wandb_run
        )
        if run_eval:
            print("Evaluating mid model...")
            evaluate_chat_model.remote(source="mid")

    if run_sft:
        print("\nStage 5/8: Supervised fine-tuning...")
        train_sft_model.remote(
            device_batch_size=device_batch_size_sft, wandb_run=wandb_run, source="mid"
        )
        if run_eval:
            print("Evaluating SFT model...")
            evaluate_chat_model.remote(source="sft")

    if run_rl:
        print("\nStage 6/8: Reinforcement learning...")
        train_rl_model.remote(wandb_run=wandb_run)
        if run_eval:
            print("Evaluating RL model...")
            evaluate_chat_model.remote(source="rl", tasks="GSM8K")

    if run_inference:
        print("\nStage 7/8: Testing inference...")
        final_source = "rl" if run_rl else "sft"
        chat_cli.remote(source=final_source, prompt="Why is the sky blue?")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Chat via CLI: modal run TrainNanochatModal.py::chat_cli --source=sft")
    print("2. Launch Web UI: modal run TrainNanochatModal.py::chat_web --source=sft")
    print("3. Run more evals: modal run TrainNanochatModal.py::evaluate_chat_model --source=sft")
    print()
```

### Quick Test Run (1 hour, $24)

```bash
modal run TrainNanochatModal.py \
  --num-data-shards=8 \
  --depth=12 \
  --run-rl=False
```

This trains a tiny model on a small dataset. Good for making sure everything works.

### Full Speedrun (4 hours, $96)

```bash
modal run TrainNanochatModal.py \
  --num-data-shards=240 \
  --depth=20 \
  --wandb-run="speedrun-$(date +%Y%m%d)" \
  --run-rl=False
```

This is the default nanochat speedrun. Produces a working ChatGPT with good performance.

### GPT-2 Grade Model (12 hours, $288)

```bash
modal run TrainNanochatModal.py \
  --num-data-shards=450 \
  --depth=26 \
  --wandb-run="gpt2-grade-$(date +%Y%m%d)" \
  --run-rl=True
```

This trains a 1B parameter model. Comparable to GPT-2 in quality.

### Running Stages Individually

You can also run each stage separately:

```bash
# 1. Download dataset
modal run TrainNanochatModal.py::download_dataset --num-shards=240

# 2. Train tokenizer
modal run TrainNanochatModal.py::train_tokenizer
modal run TrainNanochatModal.py::evaluate_tokenizer

# 3. Train base model
modal run TrainNanochatModal.py::train_base_model \
  --depth=20 \
  --wandb-run="my-run"

# 4. Evaluate base
modal run TrainNanochatModal.py::evaluate_base_model
modal run TrainNanochatModal.py::evaluate_base_loss

# 5. Midtraining
modal run TrainNanochatModal.py::train_mid_model

# 6. SFT
modal run TrainNanochatModal.py::train_sft_model

# 7. Optional RL
modal run TrainNanochatModal.py::train_rl_model

# 8. Evaluate chat model
modal run TrainNanochatModal.py::evaluate_chat_model --source=sft

# 9. Chat with it
modal run TrainNanochatModal.py::chat_cli --source=sft
modal run TrainNanochatModal.py::chat_web --source=sft
```

This is great for development - run expensive stages once, then iterate on later stages.

## Hyperparameter Tuning

### Model Size

The `depth` parameter controls model size:

```python
depth = 12  # ~200M params (quick test)
depth = 16  # ~350M params
depth = 20  # ~561M params (default speedrun)
depth = 24  # ~800M params
depth = 26  # ~1B params (GPT-2 grade)
depth = 32  # ~1.8B params (requires more GPUs)
```

**Scaling laws:**
- Parameters ≈ depth² × 64 × 12
- Training tokens ≈ parameters × 20 (Chinchilla optimal)

### Batch Size

```python
device_batch_size = 32  # Per GPU batch size

# Effective batch size = device_batch_size × num_gpus × gradient_accumulation
# nanochat uses gradient_accumulation=1 by default
```

**Guidelines:**
- 1 GPU: batch_size=16-32
- 4 GPUs: batch_size=32-64 (per device)
- 8 GPUs: batch_size=32-64 (per device)

### Learning Rate

Base model uses Muon optimizer with these defaults:
```python
learning_rate = 0.01  # Muon (much higher than Adam!)
lr_schedule = "cosine"  # Cosine decay to 10% of peak
warmup_ratio = 0.1  # Warm up for 10% of training
```

SFT and RL use AdamW:
```python
learning_rate = 3e-4  # Standard for fine-tuning
```

### Dataset Size

| Model | Parameters | Optimal Tokens | Shards Needed |
|-------|-----------|----------------|---------------|
| d12 | 200M | 4B | ~80 |
| d16 | 350M | 7B | ~140 |
| d20 | 561M | 11B | ~220 |
| d26 | 1B | 20B | ~400 |

Add 10-20% buffer because some tokens are filtered out.

## Cost Breakdown

Based on Modal pricing (~$3.50/hr for A100-80GB):

### Full Speedrun (d20, 240 shards, 4 GPUs)

| Stage | GPUs | Duration | Cost |
|-------|------|----------|------|
| Download dataset | CPU | 30 min | $0.01 |
| Train tokenizer | 1 | 45 min | $2.50 |
| Base pretraining | 4 | 5 hours | $70 |
| Midtraining | 4 | 30 min | $7 |
| SFT | 4 | 30 min | $7 |
| RL (optional) | 4 | 30 min | $7 |
| Evaluation | 4 | 30 min | $7 |
| **Total** | | **~8 hours** | **~$100** |

### Quick Test (d12, 8 shards, 4 GPUs)

| Stage | GPUs | Duration | Cost |
|-------|------|----------|------|
| Download dataset | CPU | 5 min | $0.01 |
| Train tokenizer | 1 | 30 min | $1.75 |
| Base pretraining | 4 | 30 min | $7 |
| Midtraining | 4 | 15 min | $3.50 |
| SFT | 4 | 15 min | $3.50 |
| **Total** | | **~2 hours** | **~$16** |

### Storage Costs

- Volumes: Free up to 50GB
- This project: ~30-40GB = $0/month

## Monitoring and Debugging

### Real-time Monitoring

When you run training, Modal gives you a URL:

```
View run at https://modal.com/apps/...
```

**The dashboard shows:**
- Real-time logs from all GPUs
- GPU utilization (should be 95-100%)
- Memory usage
- Cost accumulation
- Function status

### Weights & Biases

If you set up W&B, check `wandb.ai/<username>/nanochat-modal`

**Charts to watch:**

**Base training:**
- Training loss (should decrease smoothly)
- Validation loss (should track training loss)
- CORE metric (should increase over time)
- Learning rate (should follow cosine schedule)

**SFT/RL:**
- Task-specific metrics (ARC accuracy, GSM8K accuracy, etc.)
- Loss curves
- Gradient norms

### Checking Outputs

```bash
# List what's in the volume
modal volume ls nanochat-data /data/.cache/nanochat

# Check checkpoints
modal volume ls nanochat-checkpoints /data/.cache/nanochat/checkpoints

# Download something
modal volume get nanochat-checkpoints \
  /data/.cache/nanochat/checkpoints/sft \
  ./local-checkpoint
```

### GPU Utilization

For multi-GPU training, all GPUs should be utilized. If only 1 GPU shows activity:

1. Check that `NUM_GPUS_BASE` is set correctly
2. Check torchrun is spawning multiple processes
3. Check for errors in the logs

## Common Issues and Solutions

### "nanochat directory not found"

**Error:** `FileNotFoundError: nanochat`

**Fix:**
```bash
git clone https://github.com/karpathy/nanochat.git
```

Make sure it's in the same directory as `TrainNanochatModal.py`.

### CUDA Out of Memory

**Error:** `CUDA out of memory`

**Solutions:**

1. **Reduce batch size:**
   ```bash
   modal run TrainNanochatModal.py::train_base_model --device-batch-size=16
   ```

2. **Use fewer GPUs** (counter-intuitive, but each GPU needs memory):
   ```python
   NUM_GPUS_BASE = 2  # Instead of 4
   ```

3. **Reduce model size:**
   ```bash
   modal run TrainNanochatModal.py::train_base_model --depth=16
   ```

4. **Use A100-80GB** instead of A100-40GB

### Rust Compilation Fails

**Error:** Errors during rustbpe compilation

**Fix:** Usually means Rust wasn't installed correctly. The image build includes Rust installation, so this should work. If it doesn't:

1. Check the image build logs for Rust installation errors
2. Make sure you're using the `devel` CUDA image (not `runtime`)
3. Try rebuilding: `modal build TrainNanochatModal.py`

### Training Loss Not Decreasing

**Symptoms:** Loss stays flat or increases

**Checks:**

1. **Verify data is loading:**
   - Check logs for "Loaded N batches"
   - Verify dataset was downloaded

2. **Check learning rate:**
   - Might be too low (increase it)
   - Check W&B for LR schedule

3. **Model size vs dataset size:**
   - Tiny model + huge dataset = might not converge
   - Huge model + tiny dataset = will overfit

4. **Multi-GPU issues:**
   - Verify all GPUs are being used
   - Check for NCCL errors in logs

### Secrets Not Found

**Error:** `Modal Secret "nanochat-secrets" not found`

**Fix:**
```bash
modal secret create nanochat-secrets \
  WANDB_API_KEY=your_key \
  HUGGINGFACE_TOKEN=hf_your_token
```

Or use a `.env` file (script tries that first).

### Image Build Timeout

**Error:** Image build exceeds timeout

**Fix:** First build takes 15-20 minutes (Rust compilation). This is normal. If it times out:

1. Increase timeout in Modal dashboard settings
2. Or just wait - Modal caches completed layers, so re-running continues from where it failed

## Advanced Tips and Tricks

### Resume from Checkpoint

If training crashes, resume from the last checkpoint:

```python
# Nanochat automatically saves checkpoints
# Just re-run the same command - it resumes automatically
modal run TrainNanochatModal.py::train_base_model --depth=20
```

### Custom Datasets

To train on your own data:

1. **Format as FineWeb shards** (parquet files)
2. **Place in `/data/.cache/nanochat/base_data`**
3. **Update num_shards** to match your data

Or modify `nanochat/dataset.py` to load from your source.

### Experiment with Optimizers

Edit `nanochat/scripts/base_train.py` to try different optimizers:
- Muon (default, best for base training)
- AdamW (standard, works everywhere)
- Lion (newer, sometimes faster)

### Multi-Node Training

For models >2B parameters, you might want multiple machines. Modal supports this with `multi_node` parameter. Check Modal docs for details.

### Quantization

For inference, you can quantize the model:

```python
# int8 quantization
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    load_in_8bit=True
)

# int4 quantization (even faster)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    load_in_4bit=True
)
```

## Expected Results

After training the full speedrun (d20, 240 shards), you should get approximately:

### Base Model (after pretraining)

| Metric | Expected | Good | Excellent |
|--------|----------|------|-----------|
| CORE | 0.35 | 0.38 | 0.40 |
| Validation loss | 2.5 | 2.4 | 2.3 |
| Bits per byte | 1.2 | 1.15 | 1.10 |

### After SFT

| Benchmark | Expected | Good | Excellent |
|-----------|----------|------|-----------|
| ARC-Easy | 60% | 65% | 70% |
| ARC-Challenge | 30% | 35% | 40% |
| GSM8K | 40% | 50% | 60% |
| HumanEval | 15% | 20% | 25% |
| MMLU | 45% | 50% | 55% |

### After RL (optional)

| Benchmark | Expected Improvement |
|-----------|---------------------|
| GSM8K | +10-15% |
| Others | No change (RL only trains on math) |

**For context:**
- GPT-2 (1.5B): MMLU ~35%, GSM8K ~10%
- Llama 2 7B: MMLU ~45%, GSM8K ~15%
- Your d20 model (561M): MMLU ~50%, GSM8K ~40-60%

Not bad for a model you trained in 4 hours!

## What's Next?

You've built a complete LLM training pipeline from scratch. Here's what you can do next:

### 1. Scale Up

```bash
# Train a 1B parameter model
modal run TrainNanochatModal.py \
  --num-data-shards=450 \
  --depth=26 \
  --wandb-run="gpt2-grade"
```

### 2. Custom Data

- Collect your own dataset
- Train a domain-specific model
- E.g., code-only model, medical model, creative writing model

### 3. Longer Training

```bash
# Train for 2x tokens (better convergence)
modal run TrainNanochatModal.py::train_base_model \
  --depth=20 \
  --max-iterations=20000
```

### 4. Deploy for Production

Add vLLM serving (like in the Gemma tutorial):
- OpenAI-compatible API
- Auto-scaling
- High throughput

### 5. Experiment with Architecture

- Try different model widths
- Add mixture of experts
- Experiment with different attention mechanisms

### 6. Advanced Fine-tuning

- DPO (Direct Preference Optimization)
- ORPO (Odds Ratio Preference Optimization)
- Longer RL training

## Resources

- **[Nanochat GitHub](https://github.com/karpathy/nanochat)** - The original repo
- **[Andrej's Video](https://www.youtube.com/watch?v=l8pRSuU81PU)** - Building GPT from scratch
- **[Modal Documentation](https://modal.com/docs)** - Everything about Modal
- **[Modal GPU Types](https://modal.com/docs/guide/gpu)** - All available GPUs and pricing
- **[PyTorch Distributed](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)** - Understanding multi-GPU training
- **[Chinchilla Paper](https://arxiv.org/abs/2203.15556)** - Optimal compute budget scaling

## Wrapping Up

You just did what most people think requires a PhD and millions in compute: you trained a language model from absolute scratch.

Not fine-tuning. Not adapter training. Full pretraining - tokenizer, base model, everything.

And you did it in ~4 hours for ~$100. On Modal's serverless infrastructure. No cluster to manage, no DevOps nightmares, no month-long training runs.

The Unsloth tutorial showed you highly optimized fine-tuning. The Axolotl tutorial showed you production-scale multi-GPU training. This tutorial showed you the complete pipeline - everything from raw text to functioning ChatGPT.

This is the deepest you can go in understanding LLMs. You now know exactly how GPT works because you built one yourself.

The nanochat pipeline is used by researchers, educators, and companies. It's the real deal, just scaled to be accessible. And Modal made it trivial to run.

Got questions? Hit me up on Twitter [@adithya_s_k](https://x.com/adithya_s_k)!

Now go train your own ChatGPT. You have the power. 🚀
