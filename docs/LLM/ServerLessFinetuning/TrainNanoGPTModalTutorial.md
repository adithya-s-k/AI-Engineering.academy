# Training NanoGPT on Modal: Your First GPU Training Pipeline

📄 **[View Complete Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/TrainNanoGPTModal.py)**

So you want to train a GPT model but don't want to deal with the headache of setting up infrastructure? Let me show you how I do it with Modal and nanoGPT.

## Why Start with NanoGPT?

Here's the thing about nanoGPT - if you've ever wondered how GPT actually works under the hood, this is the best place to start. Andrej Karpathy built this as an educational implementation that's simple enough to understand but powerful enough to actually train real models.

It's only ~300 lines of clean PyTorch code. No abstractions hiding what's really happening. Just pure, understandable transformer training.

And honestly? It's become the go-to for anyone learning how to train language models from scratch. Plus, since it's just a regular Python repo, it's perfect for showing you how to take *any* existing codebase and run it on Modal's GPUs.

Think of this as your "Hello World" for GPU training on Modal. Once you get this working, you'll know how to run pretty much anything on serverless GPUs.

## What We're Building

We'll train a character-level GPT on Shakespeare's collected works (because why not make our model speak like the Bard?). The pipeline has three stages:

1. **Prep the data** - Download and tokenize Shakespeare (runs on CPU, saves money)
2. **Train the model** - Fire up a GPU and train our tiny GPT
3. **Generate text** - Watch our model write Shakespeare-esque text

The best part? You write all this code locally, and with one command, it runs on a beefy A100 GPU in the cloud. No SSH, no Docker, no infrastructure headaches.

## Getting Set Up

### Install Modal

First things first:

```bash
pip install modal
```

### Authenticate

Then authenticate (only need to do this once):

```bash
modal setup
```

This opens your browser and handles the OAuth flow. If you're running this in CI/CD or prefer API keys:

```bash
export MODAL_TOKEN_ID=<your_token_id>
export MODAL_TOKEN_SECRET=<your_token_secret>
```

### Clone NanoGPT

Now grab nanoGPT. We need it locally because we're going to copy it into our Modal container:

```bash
cd /path/to/your/project
git clone https://github.com/karpathy/nanoGPT.git
```

Your folder should look like this:

```
ServerLessFinetuning/
├── TrainNanoGPTModal.py    # Your Modal script (we'll create this)
└── nanoGPT/                 # The cloned repo
    ├── train.py
    ├── sample.py
    ├── data/
    └── config/
```

> **Important:** The Modal script needs to see the `nanoGPT/` folder in the same directory. When Modal builds your container image, it'll copy this entire directory into it.

## Understanding the Modal Script

Alright, let me walk you through how this works. I'll explain each piece and why it matters.

### App and Volume Setup

```python
from modal import App, Image as ModalImage, Volume

# Create the Modal app - this is your project's namespace
app = App("nanogpt-training")

# Create or get existing volume for persistent storage
# If "nanogpt-outputs" doesn't exist, Modal creates it for you
volume = Volume.from_name("nanogpt-outputs", create_if_missing=True)

# Define where to mount the volume in our containers
# This dict maps container paths to Modal volumes
VOLUME_CONFIG = {
    "/data": volume,  # Mount 'volume' at /data inside the container
}
```

So here's what's happening:

- **App**: Every Modal project needs an app. Think of it as your project container - all your functions live under this app.
- **Volume**: This is persistent storage that survives across runs. When your GPU instance shuts down (and it will, to save you money), you need somewhere to keep your model checkpoints. Volumes stick around even after your functions finish.
- **VOLUME_CONFIG**: This dict tells Modal "hey, mount this volume at `/data` in my containers". You can mount multiple volumes at different paths if you want.

The cool thing about volumes? They persist across function calls. So when you train your model and save it to `/data/out`, you can load it later in a completely different function. It just works.

### The Container Image

This is crucial - we need to tell Modal what our container should look like:

```python
NANOGPT_IMAGE = (
    # Start with a lightweight Debian base + Python 3.11
    ModalImage.debian_slim(python_version="3.11")

    # Install all the Python packages nanoGPT needs
    # Modal uses pip under the hood
    .pip_install(
        "torch",          # PyTorch for the model
        "numpy",          # Numerical operations
        "transformers",   # Hugging Face utilities
        "datasets",       # For loading datasets
        "tiktoken",       # OpenAI's tokenizer
        "tqdm",          # Progress bars
    )

    # Copy your local nanoGPT directory into the container
    # local_path: where it is on your machine
    # remote_path: where it goes in the container
    # copy=True: actually copy the files (vs just mounting)
    .add_local_dir(local_path="nanoGPT", remote_path="/root/nanoGPT", copy=True)

    # Set the working directory - all commands run from here
    .workdir("/root/nanoGPT")
)
```

Let me break this down:

1. **Start with a minimal base** - Debian slim keeps things lightweight and fast
2. **Install dependencies** - Everything nanoGPT needs to run
3. **Copy your local code** - This is the magic! `.add_local_dir()` takes the nanoGPT repo from your machine and bakes it into the container image
4. **Set working directory** - So when we run `python train.py`, we're already in `/root/nanoGPT`

The first time Modal builds this, it'll take a few minutes (installing PyTorch takes time). But Modal caches the entire image, so every subsequent run is instant. You only rebuild when you change the image definition - like adding a new package or updating nanoGPT.

## The Three-Stage Pipeline

### Stage 1: Preparing the Data

```python
@app.function(
    image=NANOGPT_IMAGE,      # Use the image we defined above
    timeout=10 * 60,           # 10 minutes timeout (in seconds)
    # Notice: No GPU specified! This runs on CPU to save money
)
def prepare_data():
    """
    Download and prep the Shakespeare dataset.
    Creates train.bin and val.bin files.
    """
    import subprocess

    print("=" * 80)
    print("PREPARING SHAKESPEARE DATASET")
    print("=" * 80)

    # Run nanoGPT's data preparation script
    # This downloads Shakespeare text and tokenizes it
    result = subprocess.run(
        ["python", "data/shakespeare_char/prepare.py"],
        capture_output=True,  # Capture output so we can print it
        text=True             # Get output as string, not bytes
    )

    # Print what happened
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # If the script failed, raise an error
    if result.returncode != 0:
        raise RuntimeError(f"Data preparation failed with code {result.returncode}")

    print("✓ Data preparation completed!")

    # Return a dict with status info
    # (Modal functions can return JSON-serializable data)
    return {"status": "completed", "dataset": "shakespeare_char"}
```

Notice what we're NOT specifying? A GPU. This function runs on CPU because we don't need a GPU just to download and tokenize text. Why pay for GPU time when we don't need it?

This function:
1. Downloads the Shakespeare text (~1MB)
2. Tokenizes it at the character level
3. Creates `train.bin` and `val.bin` files

These files stay in the container's filesystem (not the volume) since they're small and the training function needs them.

### Stage 2: Training

Now here's where the fun begins:

```python
@app.function(
    image=NANOGPT_IMAGE,           # Our container image with nanoGPT
    gpu=GPU_TYPE,                   # "a100-40gb" by default - NOW we need a GPU!
    volumes=VOLUME_CONFIG,          # Mount our volume at /data
    timeout=2 * HOURS,              # Give it 2 hours max
)
def train(
    # All hyperparameters as function arguments - makes experimenting easy!
    max_iters: int = 1000,          # How many training steps
    eval_interval: int = 500,       # Check validation loss every N steps
    batch_size: int = 64,           # Samples per batch
    block_size: int = 256,          # Context length (characters)
    n_layer: int = 6,               # Number of transformer layers
    n_head: int = 6,                # Attention heads per layer
    n_embd: int = 384,              # Hidden dimension size
    learning_rate: float = 1e-3,    # Optimizer learning rate
):
    """Train a character-level GPT on Shakespeare."""
    import subprocess
    import os
    import shutil

    print("=" * 80)
    print("TRAINING NANOGPT ON SHAKESPEARE")
    print("=" * 80)

    # Safety check: make sure data is prepared
    # If not, run prepare_data locally in this container
    if not os.path.exists("data/shakespeare_char/train.bin"):
        print("Data not found, preparing it first...")
        prepare_data.local()  # .local() runs in this same container

    # Build the training command with all our hyperparameters
    # We're basically calling: python train.py config.py --max_iters=1000 ...
    cmd = [
        "python",
        "train.py",                              # nanoGPT's training script
        "config/train_shakespeare_char.py",      # Base config file
        f"--max_iters={max_iters}",              # Override config with our params
        f"--eval_interval={eval_interval}",
        f"--batch_size={batch_size}",
        f"--block_size={block_size}",
        f"--n_layer={n_layer}",
        f"--n_head={n_head}",
        f"--n_embd={n_embd}",
        f"--learning_rate={learning_rate}",
        "--out_dir=/data/out",                   # Save to volume (persists!)
        "--dataset=shakespeare_char",            # Which dataset to use
        "--compile=False",                       # Skip torch.compile for faster startup
    ]

    print(f"Running: {' '.join(cmd)}")
    # Run the training - output streams to console in real-time
    result = subprocess.run(cmd, capture_output=False, text=True)

    # Check if training succeeded
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")

    # Copy meta.pkl (character encoding info) to output dir
    # We'll need this for sampling later
    meta_src = "data/shakespeare_char/meta.pkl"
    meta_dst = "/data/out/meta.pkl"
    if os.path.exists(meta_src):
        shutil.copy(meta_src, meta_dst)

    # THIS IS CRITICAL - persist everything to the volume!
    # Without this, your checkpoint disappears when the container shuts down
    volume.commit()

    print("\n" + "=" * 80)
    print("✓ Training completed! Model saved to /data/out")
    print("=" * 80)

    # Return info about the training run
    return {
        "status": "completed",
        "max_iters": max_iters,
        "output_dir": "/data/out",
    }
```

Few things to note here:

1. **GPU specification**: We're requesting an A100-40GB. Modal spins one up just for this function.
2. **Hyperparameters as arguments**: Makes it super easy to experiment - just pass different values when you call the function.
3. **`volume.commit()`**: This is crucial! It persists everything you wrote to `/data` back to the volume. Forget this and your checkpoint disappears when the container shuts down.
4. **Fallback data prep**: If the data isn't ready, we call `prepare_data.local()` to run it first.

The training runs just like it would locally, except it's happening on a beefy GPU in the cloud.

### Stage 3: Generating Samples

Now let's see what our model learned:

```python
@app.function(
    image=NANOGPT_IMAGE,      # Same image as training
    gpu=GPU_TYPE,              # Need GPU for inference
    volumes=VOLUME_CONFIG,     # Mount volume to access saved checkpoint
    timeout=10 * 60,           # 10 minutes should be plenty
)
def sample(
    num_samples: int = 5,           # How many texts to generate
    max_new_tokens: int = 500,      # Length of each sample
    temperature: float = 0.8,        # Randomness (0.1=boring, 1.5=wild)
    start: str = "\n",               # Starting prompt
):
    """Generate text samples from our trained model."""
    import subprocess
    import os
    import shutil

    print("=" * 80)
    print("GENERATING SAMPLES FROM TRAINED MODEL")
    print("=" * 80)

    # Sanity check: make sure the checkpoint exists
    # (It should be in the volume from training)
    if os.path.exists("/data/out/ckpt.pt"):
        print("✓ Found checkpoint: /data/out/ckpt.pt")
    else:
        print("✗ Checkpoint not found: /data/out/ckpt.pt")
        # Could raise an error here, but we'll let sample.py handle it

    # nanoGPT's sample.py looks for meta.pkl in the data directory
    # So we need to copy it from the volume to where it expects it
    os.makedirs("data/shakespeare_char", exist_ok=True)
    if os.path.exists("/data/out/meta.pkl") and not os.path.exists(
        "data/shakespeare_char/meta.pkl"
    ):
        shutil.copy("/data/out/meta.pkl", "data/shakespeare_char/meta.pkl")
        print("✓ Copied meta.pkl to expected location")

    # Build the sampling command
    cmd = [
        "python",
        "sample.py",                           # nanoGPT's sampling script
        "--out_dir=/data/out",                 # Where to find the checkpoint
        f"--num_samples={num_samples}",        # How many samples to generate
        f"--max_new_tokens={max_new_tokens}",  # Length of each sample
        f"--temperature={temperature}",         # Sampling temperature
        f"--start={start}",                    # Starting prompt
        "--compile=False",                     # Skip compilation
    ]

    print(f"Running: {' '.join(cmd)}")
    # Run sampling and capture output (we want to return it)
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print the generated text
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Check if sampling succeeded
    if result.returncode != 0:
        raise RuntimeError(f"Sampling failed with code {result.returncode}")

    print("\n" + "=" * 80)
    print("✓ Sampling completed!")
    print("=" * 80)

    # Return the generated samples
    return {"status": "completed", "samples": result.stdout}
```

This loads the checkpoint from our volume and generates text. The `temperature` parameter controls creativity - higher values mean more random (and often more interesting) outputs. At 0.1, the model plays it safe and picks the most likely next character. At 1.5, it gets wild and experimental.

## Running the Pipeline

There are two ways to run this:

### Option 1: Run Everything at Once

```python
@app.local_entrypoint()  # This decorator makes it the main entry point
def main():
    """Run the complete pipeline: data prep -> train -> sample"""
    print("🚀 Starting nanoGPT pipeline...")

    # Step 1: Prepare data (runs on CPU)
    print("📁 Preparing dataset...")
    prepare_data.remote()  # .remote() runs this on Modal's infrastructure

    # Step 2: Train model (runs on GPU)
    print("🏋️ Training model...")
    train.remote(
        max_iters=1000,      # Override default params
        eval_interval=250,
        batch_size=64
    )

    # Step 3: Generate samples (runs on GPU)
    print("✨ Generating samples...")
    sample.remote(
        num_samples=3,        # Just 3 samples
        max_new_tokens=300    # 300 characters each
    )

    print("🎉 Pipeline completed!")
```

Then just:

```bash
modal run TrainNanoGPTModal.py
```

The `.remote()` calls tell Modal to run these functions on their infrastructure, not locally. Modal handles spinning up containers, mounting volumes, and tearing everything down when done.

### Option 2: Run Steps Individually

Sometimes you want more control:

```bash
# Just prepare the data
modal run TrainNanoGPTModal.py::prepare_data

# Train with custom parameters
modal run TrainNanoGPTModal.py::train --max-iters=2000 --batch-size=128

# Generate samples
modal run TrainNanoGPTModal.py::sample
```

This is great for experimentation. You can prepare data once, then train multiple times with different hyperparameters.

## Playing with Configuration

### GPU Types

```python
GPU_TYPE = "a100-40gb"  # Default - fast and powerful
```

Your options:
- **T4**: ~$0.50/hr - Great for testing, slower training
- **L40S**: ~$1/hr - Good price/performance balance
- **A100-40GB**: ~$2.50/hr - Fast training
- **A100-80GB**: ~$3.50/hr - For larger models

For testing nanoGPT, honestly a T4 is fine. Switch to A100 when you're doing real runs.

### Training Hyperparameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `max_iters` | 1000 | How many training steps |
| `eval_interval` | 500 | Check validation loss every N steps |
| `batch_size` | 64 | Samples per batch |
| `block_size` | 256 | Context length (chars the model sees) |
| `n_layer` | 6 | Number of transformer layers |
| `n_head` | 6 | Attention heads per layer |
| `n_embd` | 384 | Hidden dimension size |
| `learning_rate` | 1e-3 | Step size for optimizer |

Want to train faster? Reduce `max_iters` to 100 while testing. Want better results? Increase `n_layer` and `n_embd` (but you'll need more memory).

### Sampling Parameters

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `num_samples` | 5 | How many texts to generate |
| `max_new_tokens` | 500 | Length of each sample |
| `temperature` | 0.8 | Randomness (0.1=boring, 1.5=wild) |
| `start` | "\n" | Starting prompt |

## Local vs Remote Execution

Inside your functions, you can choose where things run:

```python
# Run locally on your machine
prepare_data.local()

# Run remotely on Modal
prepare_data.remote()
```

**Use local when:**
- Debugging
- Testing small changes
- You have a GPU locally and want to use it

**Use remote when:**
- Production training
- You need specific GPU types
- You don't want to manage infrastructure (most of the time!)

## Adding Secrets

Need Hugging Face tokens or WandB API keys?

### Option 1: .env file

Create a `.env` file:

```bash
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_key
```

Then update your function:

```python
@app.function(
    image=NANOGPT_IMAGE,
    gpu=GPU_TYPE,
    volumes=VOLUME_CONFIG,
    secrets=[modal.Secret.from_dotenv()],  # Add this
    timeout=2 * HOURS,
)
def train(...):
    import os
    hf_token = os.environ["HF_TOKEN"]
    ...
```

### Option 2: Modal Secrets

More secure for production:

```bash
modal secret create my-secrets HF_TOKEN=xxx WANDB_API_KEY=yyy
```

```python
secrets=[modal.Secret.from_name("my-secrets")]
```

## When Things Go Wrong

### "nanoGPT directory not found"

Make sure you cloned it in the right place:

```bash
git clone https://github.com/karpathy/nanoGPT.git
ls  # Should show: nanoGPT/  TrainNanoGPTModal.py
```

### "Checkpoint not found during sampling"

Training didn't complete or you forgot `volume.commit()`. Check your training logs.

### "CUDA out of memory"

Your batch size is too big for the GPU:

```bash
modal run TrainNanoGPTModal.py::train --batch-size=32
```

Or switch to a bigger GPU by changing `GPU_TYPE = "a100-80gb"`.

### Training Taking Forever

For testing, reduce iterations:

```bash
modal run TrainNanoGPTModal.py::train --max-iters=100
```

100 iterations won't give you great results, but it'll let you verify everything works.

## Monitoring Your Training

When you run `modal run`, you'll get a URL like:

```
View run at https://modal.com/apps/...
```

Click it to see:
- Real-time logs streaming
- GPU utilization graphs
- How much you're spending
- Function status

It's actually a really nice dashboard. I keep it open while training to make sure my GPU utilization is high (means I'm not wasting money).

## Cost Optimization Tips

1. **Use CPU for data prep**: We already do this! Data preparation on CPU, training on GPU.

2. **Start with cheap GPUs**: Use T4 for testing, A100 for real runs.

3. **Set timeouts**: Don't let a buggy script run forever:
   ```python
   timeout=1 * HOURS  # Kill it after an hour
   ```

4. **Clean up old checkpoints**: Volumes are free up to 50GB, but still, no need to hoard.

## What's Next?

Now that you've got the basics down with nanoGPT, you can:

- **Experiment with hyperparameters**: Try different learning rates, model sizes
- **Use your own data**: Replace Shakespeare with your favorite books, code, whatever
- **Add WandB tracking**: Log your experiments properly
- **Try the other tutorials**: The Gemma tutorial shows production-scale fine-tuning with LoRA, and the Llama tutorial covers multi-GPU training

The pattern is always the same:
1. Write your code locally
2. Define your Modal image
3. Wrap your functions with `@app.function()`
4. Run with `modal run`

That's it. No Docker, no Kubernetes, no infrastructure headaches. Just write Python and run it on GPUs.

---

## Resources

- [NanoGPT GitHub](https://github.com/karpathy/nanoGPT) - The repo we're using
- [Modal Documentation](https://modal.com/docs) - When you want to dig deeper
- [Modal GPU Types](https://modal.com/docs/guide/gpu) - All available GPUs and pricing
- [Andrej Karpathy's Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Watch him build nanoGPT from scratch

---

Got questions? Hit me up on Twitter [@adithya_s_k](https://x.com/adithya_s_k) or check out the other tutorials in this series!
