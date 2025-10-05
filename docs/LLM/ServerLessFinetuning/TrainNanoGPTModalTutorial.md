# Training NanoGPT on Modal: Character-Level Language Modeling

📄 **[View Complete Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/TrainNanoGPTModal.py)**

This tutorial walks you through training a character-level GPT model using Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) on Modal's serverless GPU infrastructure. We'll train on the Shakespeare dataset and generate text samples.

## Why NanoGPT on Modal?

**NanoGPT** is an excellent educational implementation of GPT that's simple, clean, and fast. It's perfect for understanding transformer training from scratch.

**Modal** lets you run this training on powerful GPUs without managing infrastructure—write code locally, execute remotely.

## What You'll Learn

- Setting up Modal for GPU training
- Running existing codebases (nanoGPT) on Modal with minimal changes
- Managing data preparation, training, and inference as separate functions
- Using Modal volumes for persistent storage

---

## Prerequisites

### 1. Install Modal

```bash
pip install modal
```

### 2. Authenticate with Modal

```bash
modal setup
```

Or use API keys:

```bash
export MODAL_TOKEN_ID=<your_token_id>
export MODAL_TOKEN_SECRET=<your_token_secret>
```

### 3. Clone NanoGPT Repository

```bash
cd /path/to/your/project
git clone https://github.com/karpathy/nanoGPT.git
```

> **Note:** The Modal script expects a `nanoGPT/` directory in the same location as the Python script. The entire directory will be copied into the Modal container image.

---

## Project Structure

```
ServerLessFinetuning/
├── TrainNanoGPTModal.py    # Your Modal script
└── nanoGPT/                 # Cloned nanoGPT repository
    ├── train.py
    ├── sample.py
    ├── data/
    │   └── shakespeare_char/
    │       └── prepare.py
    └── config/
        └── train_shakespeare_char.py
```

---

## Understanding the Modal Script

### App and Volume Setup

```python
from modal import App, Image as ModalImage, Volume

app = App("nanogpt-training")
volume = Volume.from_name("nanogpt-outputs", create_if_missing=True)

VOLUME_CONFIG = {
    "/data": volume,
}
```

**What's happening:**
- `App`: Every Modal project starts with an app
- `Volume`: Persistent storage for model checkpoints and outputs
- The volume is mounted at `/data` in all functions that use it

> **💡 Important:** Volumes persist across function calls. Data stored in `/data` will be available even after your function terminates.

### Image Configuration

```python
NANOGPT_IMAGE = (
    ModalImage.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "transformers",
        "datasets",
        "tiktoken",
        "tqdm",
    )
    .add_local_dir(local_path="nanoGPT", remote_path="/root/nanoGPT", copy=True)
    .workdir("/root/nanoGPT")
)
```

**What's happening:**
- Starts with Debian slim + Python 3.11
- Installs PyTorch and dependencies
- **Copies your local `nanoGPT/` directory into the image** at `/root/nanoGPT`
- Sets working directory to `/root/nanoGPT`

> **⚠️ Critical:** The `.add_local_dir()` method copies files from your local machine into the container image. Make sure the `nanoGPT` directory exists locally before running!

---

## The Three-Stage Pipeline

### Stage 1: Data Preparation

```python
@app.function(
    image=NANOGPT_IMAGE,
    timeout=10 * 60,  # 10 minutes
)
def prepare_data():
    """
    Prepare the Shakespeare dataset for character-level training.
    This downloads the data and creates train.bin and val.bin files.
    """
    import subprocess

    print("=" * 80)
    print("PREPARING SHAKESPEARE DATASET")
    print("=" * 80)

    # Run the prepare script
    result = subprocess.run(
        ["python", "data/shakespeare_char/prepare.py"], capture_output=True, text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Data preparation failed with code {result.returncode}")

    print("✓ Data preparation completed!")
    return {"status": "completed", "dataset": "shakespeare_char"}
```

**What it does:**
- Runs on CPU (no GPU needed for data prep)
- Downloads Shakespeare text dataset
- Tokenizes and creates `train.bin` and `val.bin` files
- Files are stored in the image (not volume), so they're available to the training function

> **Note:** No GPU is specified, so this runs on CPU. Perfect for data processing tasks!

### Stage 2: Training

```python
@app.function(
    image=NANOGPT_IMAGE,
    gpu=GPU_TYPE,  # "a100-40gb" by default
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def train(
    max_iters: int = 1000,
    eval_interval: int = 500,
    batch_size: int = 64,
    block_size: int = 256,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    learning_rate: float = 1e-3,
):
    """Train a character-level GPT on Shakespeare data."""
    import subprocess
    import os
    import shutil

    print("=" * 80)
    print("TRAINING NANOGPT ON SHAKESPEARE")
    print("=" * 80)

    # Make sure data is prepared
    if not os.path.exists("data/shakespeare_char/train.bin"):
        print("Data not found, preparing it first...")
        prepare_data.local()

    # Build training command
    cmd = [
        "python",
        "train.py",
        "config/train_shakespeare_char.py",
        f"--max_iters={max_iters}",
        f"--eval_interval={eval_interval}",
        f"--batch_size={batch_size}",
        f"--block_size={block_size}",
        f"--n_layer={n_layer}",
        f"--n_head={n_head}",
        f"--n_embd={n_embd}",
        f"--learning_rate={learning_rate}",
        "--out_dir=/data/out",  # Save to volume!
        "--dataset=shakespeare_char",
        "--compile=False",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Copy meta.pkl to output directory for sampling
    meta_src = "data/shakespeare_char/meta.pkl"
    meta_dst = "/data/out/meta.pkl"
    if os.path.exists(meta_src):
        shutil.copy(meta_src, meta_dst)

    # Commit the volume to save the checkpoint
    volume.commit()

    print("\n" + "=" * 80)
    print("✓ Training completed! Model saved to /data/out")
    print("=" * 80)

    return {
        "status": "completed",
        "max_iters": max_iters,
        "output_dir": "/data/out",
    }
```

**Key points:**
- Runs on A100 GPU (configurable via `GPU_TYPE`)
- Uses the `/data` volume to persist model checkpoints
- If data isn't prepared, calls `prepare_data.local()` to run it
- **`volume.commit()`** is crucial—it saves checkpoint to persistent storage
- All hyperparameters are exposed as function arguments

> **💾 Volume Commit:** Always call `volume.commit()` after writing files you want to persist!

### Stage 3: Sampling (Inference)

```python
@app.function(
    image=NANOGPT_IMAGE,
    gpu=GPU_TYPE,
    volumes=VOLUME_CONFIG,
    timeout=10 * 60,
)
def sample(
    num_samples: int = 5,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    start: str = "\n",
):
    """Generate text samples from the trained model."""
    import subprocess
    import os
    import shutil

    print("=" * 80)
    print("GENERATING SAMPLES FROM TRAINED MODEL")
    print("=" * 80)

    # Check if model files exist
    if os.path.exists("/data/out/ckpt.pt"):
        print("✓ Found checkpoint: /data/out/ckpt.pt")
    else:
        print("✗ Checkpoint not found: /data/out/ckpt.pt")

    # Ensure meta.pkl is in the data directory
    os.makedirs("data/shakespeare_char", exist_ok=True)
    if os.path.exists("/data/out/meta.pkl") and not os.path.exists(
        "data/shakespeare_char/meta.pkl"
    ):
        shutil.copy("/data/out/meta.pkl", "data/shakespeare_char/meta.pkl")

    # Build sampling command
    cmd = [
        "python",
        "sample.py",
        "--out_dir=/data/out",  # Read from volume
        f"--num_samples={num_samples}",
        f"--max_new_tokens={max_new_tokens}",
        f"--temperature={temperature}",
        f"--start={start}",
        "--compile=False",
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Sampling failed with code {result.returncode}")

    print("\n" + "=" * 80)
    print("✓ Sampling completed!")
    print("=" * 80)

    return {"status": "completed", "samples": result.stdout}
```

**Key points:**
- Loads checkpoint from `/data/out` (the volume)
- Copies `meta.pkl` for character encoding
- Generates samples using trained model
- Temperature controls randomness (higher = more creative)

---

## Running the Pipeline

### Option 1: Run Everything (Full Pipeline)

```python
@app.local_entrypoint()
def main():
    """Run the complete pipeline: prepare data -> train -> sample"""
    print("🚀 Starting nanoGPT pipeline...")

    # Prepare data
    print("📁 Preparing dataset...")
    prepare_data.remote()

    # Train model
    print("🏋️ Training model...")
    train.remote(max_iters=1000, eval_interval=250, batch_size=64)

    # Generate samples
    print("✨ Generating samples...")
    sample.remote(num_samples=3, max_new_tokens=300)

    print("🎉 Pipeline completed!")
```

**Run it:**

```bash
modal run TrainNanoGPTModal.py
```

> **Note:** `.remote()` executes the function on Modal's infrastructure. The full pipeline runs sequentially.

### Option 2: Run Individual Steps

**Prepare data only:**
```bash
modal run TrainNanoGPTModal.py::prepare_data
```

**Train only:**
```bash
modal run TrainNanoGPTModal.py::train
```

**Sample only:**
```bash
modal run TrainNanoGPTModal.py::sample
```

**Custom training parameters:**
```bash
modal run TrainNanoGPTModal.py::train --max-iters=2000 --batch-size=128 --learning-rate=0.0003
```

---

## Configuration Options

### GPU Type

```python
GPU_TYPE = "a100-40gb"  # Options: "a100-40gb", "a100-80gb", "l40s", "t4"
```

- **T4**: Budget option, good for testing
- **L40S**: Great price/performance
- **A100-40GB**: Fast training
- **A100-80GB**: For larger models

### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iters` | 1000 | Number of training iterations |
| `eval_interval` | 500 | Evaluate every N steps |
| `batch_size` | 64 | Batch size for training |
| `block_size` | 256 | Context length (max sequence length) |
| `n_layer` | 6 | Number of transformer layers |
| `n_head` | 6 | Number of attention heads |
| `n_embd` | 384 | Embedding dimension |
| `learning_rate` | 1e-3 | Learning rate |

### Sampling Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_samples` | 5 | How many text samples to generate |
| `max_new_tokens` | 500 | Length of each sample |
| `temperature` | 0.8 | Sampling temperature (0.1-1.5) |
| `start` | "\n" | Starting prompt |

---

## Running Locally vs. Modal

### Local Execution

```python
# Inside the function, call .local() instead of .remote()
prepare_data.local()  # Runs on your local machine
```

### Remote Execution (Modal)

```python
prepare_data.remote()  # Runs on Modal's infrastructure
```

> **When to use local:**
> - Debugging
> - Testing small changes
> - When you have local GPU
>
> **When to use remote:**
> - Production training
> - Need specific GPU types
> - Don't want to manage infrastructure

---

## Environment Variables and Secrets

If you need Hugging Face access or other secrets:

### Create a .env file

```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

### Update the function decorator

```python
@app.function(
    image=NANOGPT_IMAGE,
    gpu=GPU_TYPE,
    volumes=VOLUME_CONFIG,
    secrets=[modal.Secret.from_dotenv()],  # Add this line
    timeout=2 * HOURS,
)
def train(...):
    import os
    # Now you can access: os.environ["HF_TOKEN"]
    ...
```

### Or use Modal Secrets

```bash
modal secret create my-secrets HF_TOKEN=xxx WANDB_API_KEY=yyy
```

```python
secrets=[modal.Secret.from_name("my-secrets")]
```

---

## Common Issues and Solutions

### Issue 1: "nanoGPT directory not found"

**Solution:** Make sure you've cloned nanoGPT in the same directory:

```bash
git clone https://github.com/karpathy/nanoGPT.git
ls  # Should show: nanoGPT/  TrainNanoGPTModal.py
```

### Issue 2: "Checkpoint not found during sampling"

**Solution:** Make sure training completed successfully and you called `volume.commit()`. Check if `/data/out/ckpt.pt` exists in the volume.

### Issue 3: "CUDA out of memory"

**Solution:** Reduce batch size or switch to larger GPU:

```bash
modal run TrainNanoGPTModal.py::train --batch-size=32
```

Or change `GPU_TYPE = "a100-80gb"` in the script.

### Issue 4: Training taking too long

**Solution:** Reduce iterations for testing:

```bash
modal run TrainNanoGPTModal.py::train --max-iters=100
```

---

## Monitoring Training

Modal provides a web UI to monitor your functions:

1. After running `modal run`, you'll see a URL like:
   ```
   View run at https://modal.com/apps/...
   ```

2. Click it to see:
   - Real-time logs
   - GPU utilization
   - Cost tracking
   - Function status

---

## Cost Optimization Tips

1. **Use smaller GPUs for testing:**
   ```python
   GPU_TYPE = "t4"  # Cheapest option
   ```

2. **Set appropriate timeouts:**
   ```python
   timeout=1 * HOURS  # Don't let it run indefinitely
   ```

3. **Use CPU for data prep:**
   - Data preparation doesn't need GPU
   - Training and sampling do

4. **Monitor volume usage:**
   - Volumes are free up to a limit
   - Clean up old checkpoints regularly

---

## Complete Script

```python
"""
Simple Modal script to run nanoGPT training on serverless GPUs.

This demonstrates how you can take a local repository (nanoGPT) and run it
on Modal with minimal changes - just copy the code into the image and run!

Usage:
    # Prepare Shakespeare dataset and train a small GPT
    modal run TrainNanoGPTModal.py

    # Or run individual steps:
    modal run TrainNanoGPTModal.py::prepare_data
    modal run TrainNanoGPTModal.py::train
    modal run TrainNanoGPTModal.py::sample
"""

from modal import App, Image as ModalImage, Volume

# =============================================================================
# CONFIGURATION
# =============================================================================

HOURS = 60 * 60
GPU_TYPE = "a100-40gb"  # Can be: a100-40gb, a100-80gb, l40s, t4, etc.

# =============================================================================
# MODAL APP AND VOLUME SETUP
# =============================================================================

app = App("nanogpt-training")
volume = Volume.from_name("nanogpt-outputs", create_if_missing=True)

VOLUME_CONFIG = {
    "/data": volume,
}

# =============================================================================
# IMAGE SETUP - Copy local nanoGPT repo into the image
# =============================================================================

# Simple approach: copy the entire nanoGPT directory into the image
NANOGPT_IMAGE = (
    ModalImage.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "numpy",
        "transformers",
        "datasets",
        "tiktoken",
        "tqdm",
    )
    # Copy the nanoGPT directory from local filesystem into the image
    # copy=True because we have .workdir() after this
    .add_local_dir(local_path="nanoGPT", remote_path="/root/nanoGPT", copy=True)
    .workdir("/root/nanoGPT")
)

# =============================================================================
# DATA PREPARATION FUNCTION
# =============================================================================


@app.function(
    image=NANOGPT_IMAGE,
    timeout=10 * 60,  # 10 minutes
)
def prepare_data():
    """
    Prepare the Shakespeare dataset for character-level training.
    This downloads the data and creates train.bin and val.bin files.
    """
    import subprocess

    print("=" * 80)
    print("PREPARING SHAKESPEARE DATASET")
    print("=" * 80)

    # Run the prepare script
    result = subprocess.run(
        ["python", "data/shakespeare_char/prepare.py"], capture_output=True, text=True
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Data preparation failed with code {result.returncode}")

    print("✓ Data preparation completed!")
    return {"status": "completed", "dataset": "shakespeare_char"}


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


@app.function(
    image=NANOGPT_IMAGE,
    gpu=GPU_TYPE,
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
)
def train(
    max_iters: int = 1000,
    eval_interval: int = 500,
    batch_size: int = 64,
    block_size: int = 256,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    learning_rate: float = 1e-3,
):
    """
    Train a character-level GPT on Shakespeare data.

    This runs the nanoGPT training script with customizable hyperparameters.
    The trained model checkpoint will be saved to the Modal volume.

    Args:
        max_iters: Number of training iterations
        eval_interval: How often to evaluate
        batch_size: Batch size for training
        block_size: Context length
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension
        learning_rate: Learning rate
    """
    import subprocess
    import os

    print("=" * 80)
    print("TRAINING NANOGPT ON SHAKESPEARE")
    print("=" * 80)
    print(f"Max iterations: {max_iters}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Layers: {n_layer}, Heads: {n_head}, Embedding: {n_embd}")
    print("=" * 80)

    # Make sure data is prepared
    if not os.path.exists("data/shakespeare_char/train.bin"):
        print("Data not found, preparing it first...")
        prepare_data.local()

    # Build training command with arguments
    cmd = [
        "python",
        "train.py",
        "config/train_shakespeare_char.py",
        f"--max_iters={max_iters}",
        f"--eval_interval={eval_interval}",
        f"--batch_size={batch_size}",
        f"--block_size={block_size}",
        f"--n_layer={n_layer}",
        f"--n_head={n_head}",
        f"--n_embd={n_embd}",
        f"--learning_rate={learning_rate}",
        "--out_dir=/data/out",  # Save outputs to volume
        "--dataset=shakespeare_char",  # Important: tells sample.py where to find meta.pkl
        "--compile=False",  # Disable compilation for faster startup
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    # Run training
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Copy meta.pkl to output directory for sampling
    import shutil

    meta_src = "data/shakespeare_char/meta.pkl"
    meta_dst = "/data/out/meta.pkl"
    if os.path.exists(meta_src):
        shutil.copy(meta_src, meta_dst)
        print(f"✓ Copied {meta_src} to {meta_dst}")

    # Commit the volume to save the checkpoint
    volume.commit()

    print("\n" + "=" * 80)
    print("✓ Training completed! Model saved to /data/out")
    print("=" * 80)

    return {
        "status": "completed",
        "max_iters": max_iters,
        "output_dir": "/data/out",
    }


# =============================================================================
# SAMPLING FUNCTION
# =============================================================================


@app.function(
    image=NANOGPT_IMAGE,
    gpu=GPU_TYPE,
    volumes=VOLUME_CONFIG,
    timeout=10 * 60,
)
def sample(
    num_samples: int = 5,
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    start: str = "\n",
):
    """
    Generate text samples from the trained model.

    Args:
        num_samples: Number of samples to generate
        max_new_tokens: Length of each sample
        temperature: Sampling temperature (higher = more random)
        start: Starting prompt for generation
    """
    import subprocess
    import os

    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    print("=" * 80)
    print("GENERATING SAMPLES FROM TRAINED MODEL")
    print("=" * 80)
    print(f"Num samples: {num_samples}")
    print(f"Max tokens: {max_new_tokens}")
    print(f"Temperature: {temperature}")
    print(f"Start prompt: {repr(start)}")
    print("=" * 80)

    # Check if model files exist
    if os.path.exists("/data/out/ckpt.pt"):
        print("✓ Found checkpoint: /data/out/ckpt.pt")
    else:
        print("✗ Checkpoint not found: /data/out/ckpt.pt")

    if os.path.exists("/data/out/meta.pkl"):
        print("✓ Found meta file: /data/out/meta.pkl")
    else:
        print("✗ Meta file not found: /data/out/meta.pkl")
        print("  Sampling will use GPT-2 encoding which will fail!")

    print()

    # Ensure meta.pkl exists in the data directory for sample.py to find
    # sample.py looks for meta.pkl in data/{dataset}/meta.pkl first, then falls back to out_dir
    import shutil

    os.makedirs("data/shakespeare_char", exist_ok=True)

    # Copy meta.pkl from volume to data directory if it exists
    if os.path.exists("/data/out/meta.pkl") and not os.path.exists(
        "data/shakespeare_char/meta.pkl"
    ):
        shutil.copy("/data/out/meta.pkl", "data/shakespeare_char/meta.pkl")
        print("✓ Copied meta.pkl to data/shakespeare_char/")

    # Build sampling command
    cmd = [
        "python",
        "sample.py",
        "--out_dir=/data/out",  # Read model from volume
        f"--num_samples={num_samples}",
        f"--max_new_tokens={max_new_tokens}",
        f"--temperature={temperature}",
        f"--start={start}",
        "--compile=False",
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    # Run sampling
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    if result.returncode != 0:
        raise RuntimeError(f"Sampling failed with code {result.returncode}")

    print("\n" + "=" * 80)
    print("✓ Sampling completed!")
    print("=" * 80)

    return {"status": "completed", "samples": result.stdout}


# =============================================================================
# LOCAL ENTRYPOINT - Run everything in sequence
# =============================================================================


@app.local_entrypoint()
def main():
    """Run the complete pipeline: prepare data -> train -> sample"""
    print("🚀 Starting nanoGPT pipeline...")

    # Prepare data
    print("📁 Preparing dataset...")
    prepare_data.remote()

    # Train model
    print("🏋️ Training model...")
    train.remote(max_iters=1000, eval_interval=250, batch_size=64)

    # Generate samples
    print("✨ Generating samples...")
    sample.remote(num_samples=3, max_new_tokens=300)

    print("🎉 Pipeline completed!")
```

---

## Next Steps

- **Experiment with hyperparameters:** Try different learning rates, model sizes
- **Use your own dataset:** Replace Shakespeare with custom text data
- **Monitor with WandB:** Add Weights & Biases integration for experiment tracking
- **Try other Modal examples:** Check out the Gemma and Llama tutorials for production-scale fine-tuning

---

## Resources

- [NanoGPT GitHub](https://github.com/karpathy/nanoGPT)
- [Modal Documentation](https://modal.com/docs)
- [Modal GPU Types](https://modal.com/docs/guide/gpu)
- [Andrej Karpathy's YouTube Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)
