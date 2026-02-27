"""
Simple Modal script to run nanoGPT training on serverless GPUs.

This demonstrates how you can take a local repository (nanoGPT) and run it
on Modal with minimal changes - just copy the code into the image and run!

Usage:
    # Prepare Shakespeare dataset and train a small GPT
    modal run FinetuneNanoGPT.py

    # Or run individual steps:
    modal run FinetuneNanoGPT.py::prepare_data
    modal run FinetuneNanoGPT.py::train
    modal run FinetuneNanoGPT.py::sample
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
