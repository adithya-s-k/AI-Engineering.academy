# Fine-tuning Gemma 3-4B with Unsloth on Modal: Production-Ready Vision-Language Training

📄 **[View Complete Python Script](https://github.com/adithya-s-k/AI-Engineering.academy/blob/main/docs/LLM/ServerLessFinetuning/FinetuneGemmaUnslothModal.py)**

🔗 **[Original Unsloth Colab Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision.ipynb)**

So you've mastered the basics with nanoGPT. Now let's level up and build a production-grade ML pipeline - we're talking dataset management, LoRA fine-tuning, model evaluation, and deployment. All on Modal's serverless infrastructure.

We'll fine-tune Google's Gemma 3-4B vision model to read LaTeX equations from images. By the end, you'll have a fully deployed API that can look at a math equation and spit out the LaTeX code for it.

## Why Unsloth?

Here's the thing - training large language models is expensive and slow. Unsloth changes that game completely.

I discovered Unsloth when I was trying to fine-tune Llama models and getting frustrated with how slow everything was. Then I found this library that claimed "2x faster training" and I was skeptical. But holy shit, it actually delivers.

**What makes Unsloth special:**
- **2-5x faster training** than standard Hugging Face Transformers (no joke, you'll see the difference)
- **60-80% less memory usage** - fits bigger models on smaller GPUs
- **Built-in LoRA and QLoRA support** - efficient fine-tuning out of the box
- **Optimized kernels** for vision-language models like Gemma, Llama, Qwen
- **Drop-in replacement** for Hugging Face - same API, just faster

The original Colab notebook from Unsloth shows you how to do this on a single GPU. We're taking that exact workflow and making it run on Modal, so you can:
- Train on any GPU type (A100-80GB? Sure!)
- Separate data prep from training (save money)
- Deploy with vLLM for high-throughput inference
- Scale to production without changing your code

Think of this as "the Unsloth Colab notebook, but productionized".

## What We're Building

This isn't just a training script. We're building a complete ML pipeline that handles everything from data to deployment:

1. **Download datasets** (on CPU, because why waste GPU money?)
2. **Download and cache models** (one time cost, reuse forever)
3. **Fine-tune with LoRA** (the actual training)
4. **Evaluate performance** (with real metrics, not vibes)
5. **Deploy with vLLM** (production-ready serving with auto-scaling)

The cool part? Each stage is independent. Screw up training? Just re-run that step. Want to evaluate a different checkpoint? Easy.

Here's what the flow looks like:

```
┌─────────────────┐
│  Download Data  │  (CPU - $0.00001/hr)
└────────┬────────┘
         │
┌────────▼────────┐
│ Download Model  │  (L40S - $1/hr, one time)
└────────┬────────┘
         │
┌────────▼────────┐
│   Fine-tune     │  (A100-80GB - $3.50/hr)
│   with LoRA     │
└────────┬────────┘
         │
┌────────▼────────┐
│  Export/Merge   │  (A100-80GB - ~10 min)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼──────┐
│ Serve │ │ Evaluate│  (Both use the deployed model)
│ vLLM  │ │  Model  │
└───────┘ └─────────┘
```

## Getting Started

### Install Modal

Same as before:

```bash
pip install modal
```

### Authenticate

```bash
modal setup
```

Or use API keys:

```bash
export MODAL_TOKEN_ID=<your_token_id>
export MODAL_TOKEN_SECRET=<your_token_secret>
```

### Set Up Your Secrets

This time we actually need some secrets because we're downloading from Hugging Face and (optionally) logging to Weights & Biases.

**You'll need:**
- A Hugging Face token (get it from [hf.co/settings/tokens](https://huggingface.co/settings/tokens))
- A Weights & Biases API key (optional but highly recommended - get it from [wandb.ai/authorize](https://wandb.ai/authorize))

#### Option 1: .env file (easiest for local development)

Create a `.env` file in your project:

```bash
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxx
```

#### Option 2: Modal Secrets (better for production)

```bash
modal secret create secrets-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxx \
  WANDB_API_KEY=xxxxxxxxxxxxx
```

> **Note:** The script looks for a secret named `secrets-hf-wandb`. If you use a different name, just update the code where it says `Secret.from_name("secrets-hf-wandb")`.

### Project Structure

Beautiful thing about this? It's just one file:

```
ServerLessFinetuning/
├── FinetuneGemmaUnslothModal.py    # Everything lives here
└── .env                             # Optional: your secrets
```

No cloning repos, no juggling dependencies. Just one Python file that does it all.

## Understanding the Pipeline

Let's break down what we're building. This is a **production-grade** ML pipeline with 6 independent stages. You can run any stage separately, which is huge for development and debugging.

### Stage Overview

1. **Dataset Download** - Grab the LaTeX OCR dataset (images of equations + LaTeX code)
2. **Model Download** - Download Gemma 3-4B and cache it (so we don't re-download every time)
3. **LoRA Fine-tuning** - Train adapters to teach Gemma to read equations
4. **Model Export** - Merge LoRA adapters into the base model (makes deployment easier)
5. **vLLM Serving** - Deploy as an OpenAI-compatible API with auto-scaling
6. **Evaluation** - Measure accuracy with real metrics (character error rate, exact match, etc.)

Each stage saves its outputs to a Modal volume, so the next stage can pick up where the last one left off.

## Configuration and Setup

Alright, let's dive into the code. I'll walk you through each piece and explain why it matters.

### App, Volume, and Secrets

```python
from modal import App, Image as ModalImage, Volume, Secret

# Create the Modal app - this is our project namespace
app = App("Finetuned_Gemma_3_4b_it")

# Create persistent storage - everything goes here
# Models, datasets, checkpoints, evaluation results - all in one volume
exp_volume = Volume.from_name("Finetuned_Gemma_3_4b_it", create_if_missing=True)

# Mount the volume at /data in all our containers
VOLUME_CONFIG = {
    "/data": exp_volume,  # Single volume for the entire experiment
}

# Load secrets for Hugging Face and Weights & Biases
# This injects HUGGINGFACE_TOKEN and WANDB_API_KEY as environment variables
huggingface_secret = Secret.from_name("secrets-hf-wandb")
```

**What's happening here:**

- **Volume strategy**: I use a single volume for the entire experiment. Models in `/data/.cache`, checkpoints in `/data/Finetuned_Gemma_3_4b_it`, datasets in `/data/.cache`. Keeps everything organized and makes debugging easier.
- **Secrets**: Modal injects these as environment variables. So inside our functions, we can just do `os.environ["HUGGINGFACE_TOKEN"]`.

### Configuration Constants

```python
# Time constants
HOURS = 60 * 60  # Makes timeouts more readable

# Model configuration
BASE_MODEL_NAME = "unsloth/gemma-3-4b-it"  # Unsloth's optimized Gemma
WANDB_PROJECT_DEFAULT = "GemmaFinetuning"   # W&B project name
OUTPUT_DIR_DEFAULT = "/data/Finetuned_Gemma_3_4b_it"  # Where to save checkpoints
```

These constants make it easy to swap models or change output directories. Want to try Llama instead? Just change `BASE_MODEL_NAME`.

## Building the Training Image

This is where things get interesting. We need a container with CUDA, PyTorch, Unsloth, and a bunch of other stuff.

### Why CUDA "devel"?

```python
CUDA_VERSION = "12.8.1"     # Latest CUDA version
CUDA_FLAVOR = "devel"        # "devel" includes nvcc compiler
CUDA_OS = "ubuntu24.04"      # Ubuntu 24.04 LTS
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"
```

Here's the deal: some packages like `flash-attn` and `triton` need to compile CUDA code during installation. If you use the `runtime` image, you'll get cryptic errors about missing `nvcc`. Trust me, I learned this the hard way.

The `devel` image includes the full CUDA toolkit with the compiler. It's bigger, but it Just Works™.

### Complete Image Definition

```python
FINETUNING_GPU_IMAGE = (
    # Start with NVIDIA's official CUDA image
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")

    # Install system dependencies
    # git: for cloning repos if needed
    # build-essential: gcc, make, etc. for compiling Python extensions
    .apt_install("git", "build-essential")

    # Install PyTorch first (required by most other packages)
    # Using uv for faster installs (it's like pip but 10-100x faster)
    .uv_pip_install(["torch", "torchvision", "torchaudio"])

    # Now install the ML ecosystem
    .uv_pip_install([
        # === Unsloth core ===
        "unsloth",              # The star of the show - optimized training
        "unsloth_zoo",          # Pre-configured models

        # === Quantization and efficiency ===
        "bitsandbytes",         # 8-bit optimizers, quantization
        "accelerate",           # Multi-GPU support, mixed precision
        "xformers",             # Memory-efficient attention
        "peft",                 # LoRA and other parameter-efficient methods
        "trl",                  # Transformer Reinforcement Learning
        "triton",               # GPU kernel language (used by flash-attn)
        "cut_cross_entropy",    # Optimized loss computation

        # === Transformers ecosystem ===
        "transformers",         # Hugging Face transformers
        "timm",                 # Vision model utilities

        # === Training tools ===
        "wandb",                # Experiment tracking (highly recommend!)
        "weave",                # W&B's LLM eval framework
        "deepspeed",            # For multi-GPU training (optional here)

        # === Evaluation metrics ===
        "nltk",                 # NLP toolkit
        "rouge_score",          # ROUGE metrics
        "bert_score",           # BERTScore
        "jiwer",                # Word/Character Error Rate
        "scikit-learn",         # General ML utilities

        # === Utilities ===
        "pillow",               # Image processing
        "opencv-python-headless",  # More image processing
        "gradio",               # Quick UI demos
        "hf_transfer",          # Faster Hugging Face downloads
    ])

    # Set environment variables
    .env({
        # Enable fast multi-threaded downloads from Hugging Face
        # This can be 5-10x faster for large models!
        "HF_HUB_ENABLE_HF_TRANSFER": "1",

        # Cache everything in the volume (so it persists)
        # This means we download models once, use them forever
        "HF_HOME": "/data/.cache",
    })
)
```

**Key points:**

1. **uv_pip_install**: Modal uses `uv` under the hood, which is stupid fast. Installing 20+ packages takes like 2 minutes instead of 10.

2. **HF_HUB_ENABLE_HF_TRANSFER**: This enables Hugging Face's `hf_transfer` library which downloads models in parallel. For a 16GB model, this can cut download time from 10 minutes to 2 minutes.

3. **HF_HOME in volume**: By setting this to `/data/.cache`, all Hugging Face downloads get cached in our volume. Download a model once, use it in all future runs.

> **⏰ Build time warning**: The first time you run this, Modal will build the image. It takes 10-15 minutes because of all the compilation (flash-attn especially). Grab a coffee. But here's the magic - Modal caches the image. Every subsequent run? Instant.

## Stage 1: Downloading Datasets

Let's start with data. We're using Unsloth's LaTeX OCR dataset - images of math equations paired with their LaTeX code.

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,     # Our big image with all dependencies
    volumes=VOLUME_CONFIG,           # Mount /data volume
    secrets=[huggingface_secret],   # Inject HF token
    timeout=24 * HOURS,              # Give it up to 24 hours (large datasets)
    # Notice: No GPU! This runs on CPU to save money
)
def download_datasets(
    dataset_name: str = "unsloth/LaTeX_OCR",  # HuggingFace dataset ID
    split: str = "train",                      # Which split to download
    cache_dir: str = "/data/.cache",           # Where to cache it
):
    """
    Download and cache a dataset from Hugging Face.

    Runs on CPU (no GPU wasted on downloading files).
    Dataset gets cached in the volume, so we only download once.
    """
    from datasets import load_dataset
    import os

    # Set HF token from our secret
    # Modal injects HUGGINGFACE_TOKEN from the secret we passed in
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Make sure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading {dataset_name} ({split} split)...")
    print(f"Cache dir: {cache_dir}")

    # Download the dataset
    # cache_dir tells it to save in our volume (persists across runs)
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    # Print some info
    print("\n✓ Dataset loaded successfully!")
    print(f"  - Name: {dataset_name}")
    print(f"  - Split: {split}")
    print(f"  - Number of samples: {len(dataset)}")
    print(f"  - Features: {dataset.features}")

    # CRITICAL: Commit changes to the volume
    # This persists the downloaded data
    exp_volume.commit()

    # Return metadata
    return {
        "status": "completed",
        "dataset_name": dataset_name,
        "num_samples": len(dataset),
    }
```

**Why download separately?**

You might be thinking "why not just download during training?" Here's why this is better:

1. **No GPU waste**: Downloading files doesn't need a GPU. Why pay $3.50/hr for an A100 when a CPU costs pennies?
2. **Faster iteration**: Download once, train many times with different hyperparameters
3. **Debugging**: If download fails, you know immediately. Not after 10 minutes of training setup.

**Running it:**

```bash
# Download the default dataset (LaTeX OCR)
modal run FinetuneGemmaUnslothModal.py::download_datasets

# Or download a custom dataset
modal run FinetuneGemmaUnslothModal.py::download_datasets \
  --dataset-name="your-username/your-dataset" \
  --split="train"
```

The first time you run this, it downloads and caches the dataset. Second time? Instant, because it's already in the volume.

## Stage 2: Downloading Models

Same idea as datasets - download once, use forever.

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,
    gpu="l40s:1",                   # Use a cheap GPU (L40S is ~$1/hr)
    volumes=VOLUME_CONFIG,           # Mount our volume
    secrets=[huggingface_secret],   # Need HF token for model access
    timeout=24 * HOURS,
)
def download_models(
    model_name: str = BASE_MODEL_NAME,      # "unsloth/gemma-3-4b-it"
    cache_dir: str = "/data/.cache",        # Cache in volume
):
    """
    Download and cache the base model using Unsloth's FastVisionModel.

    Why L40S GPU? Some models need a GPU just to load (for safety checks, etc.)
    L40S is cheaper than A100, perfect for this one-time download.
    """
    from unsloth import FastVisionModel
    import os
    import torch

    # Set HF token
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    print(f"Downloading model: {model_name}")
    print(f"Cache dir: {cache_dir}")

    # Load the model with Unsloth's optimized loader
    # This downloads and caches the model weights
    model, processor = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=False,                    # Full precision for now
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        max_seq_length=8000,                    # Max context length
        dtype=torch.bfloat16,                   # Use bfloat16 (good balance)
    )

    print(f"\n✓ Model downloaded and cached!")
    print(f"  - Model: {model_name}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Cache: {cache_dir}")

    # Commit to volume
    exp_volume.commit()

    return {
        "status": "completed",
        "model_name": model_name,
        "cache_dir": cache_dir,
    }
```

**Why use a GPU for downloading?**

Some models (especially gated ones like Gemma) run initialization code that requires a GPU. It's annoying, but that's how it is. We use an L40S because it's cheap (~$1/hr) and we only do this once.

**Run it:**

```bash
modal run FinetuneGemmaUnslothModal.py::download_models
```

First run downloads ~16GB (takes a few minutes with `hf_transfer`). Every subsequent run? Instant.

## Stage 3: Fine-tuning with LoRA

Alright, here's where the magic happens. We're going to fine-tune Gemma 3-4B to read LaTeX equations from images.

### GPU Configuration

```python
TRAIN_GPU = "a100-80gb"    # For 4B vision models, A100-80GB is ideal
NUM_GPUS = 1                # Unsloth is optimized for single-GPU
TRAINING_GPU_CONFIG = f"{TRAIN_GPU}:{NUM_GPUS}"
```

**Why A100-80GB?**
- Vision-language models are memory-hungry (images take a lot of VRAM)
- 4B model + images + gradients = needs ~40-60GB
- A100-40GB might OOM, A100-80GB is comfortable

**Why single GPU?**
- Unsloth is insanely optimized for single-GPU training
- Multi-GPU adds communication overhead
- For most fine-tuning, single A100 is faster than 2-4 smaller GPUs

### The Training Function

This is a big one, so I'll break it into pieces:

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret, Secret.from_dotenv()],  # Both Modal secrets and .env
    gpu=TRAINING_GPU_CONFIG,                              # "a100-80gb:1"
    timeout=24 * HOURS,                                   # Long timeout for big datasets
)
def fine_tune_unsloth(
    # Model and dataset config
    model_path: str = BASE_MODEL_NAME,                 # Which model to fine-tune
    dataset_name: str = "unsloth/LaTeX_OCR",          # Which dataset to use
    dataset_split: str = "train",                      # Which split
    output_dir: str = OUTPUT_DIR_DEFAULT,              # Where to save checkpoints
    hub_id: str = None,                                # Push to HF Hub? (optional)
    max_samples: int = None,                           # Limit dataset (for testing)

    # LoRA hyperparameters
    lora_r: int = 32,                                  # LoRA rank (higher = more capacity)
    lora_alpha: int = 64,                              # LoRA scaling (usually 2x rank)
    lora_dropout: float = 0.0,                         # Dropout in LoRA layers

    # Training hyperparameters
    per_device_train_batch_size: int = 4,              # Batch size per GPU
    gradient_accumulation_steps: int = 4,              # Effective batch = 4 * 4 = 16
    num_train_epochs: int = 1,                         # How many epochs
    learning_rate: float = 3e-4,                       # Learning rate
    warmup_ratio: float = 0.2,                         # Warmup 20% of steps
    max_seq_length: int = 8000,                        # Max tokens per sample

    # Checkpointing
    save_strategy: str = "steps",                      # Save by steps (not epochs)
    save_steps: int = 250,                             # Save every 250 steps
    save_total_limit: int = 20,                        # Keep only 20 checkpoints
    logging_steps: int = 10,                           # Log every 10 steps

    # Weights & Biases
    wandb_project: str = WANDB_PROJECT_DEFAULT,        # W&B project name
    wandb_run_name: str = None,                        # W&B run name (auto-generated)
):
    """
    Fine-tune Gemma 3-4B vision model with LoRA using Unsloth.

    This is based on Unsloth's Colab notebook but productionized for Modal.
    """
    from unsloth import FastVisionModel, get_chat_template
    from unsloth.trainer import UnslothVisionDataCollator
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    import torch
    import os
    from datetime import datetime
```

Let me continue with the rest of the training function with detailed comments:

```python
    # === Environment Setup ===
    print("=" * 80)
    print("SETTING UP TRAINING ENVIRONMENT")
    print("=" * 80)

    # Set up authentication tokens
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["WANDB_PROJECT"] = wandb_project

    # Auto-generate W&B run name if not provided
    # Format: finetune_gemma-3-4b-it_20250110_143022
    if wandb_run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_path.split("/")[-1]  # Extract "gemma-3-4b-it" from path
        wandb_run_name = f"finetune_{model_short}_{timestamp}"

    os.environ["WANDB_RUN_NAME"] = wandb_run_name

    # Memory optimization: only use GPU 0
    # (In single-GPU setup, this prevents memory fragmentation)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Disable torch compile (can cause issues with some models)
    torch._dynamo.config.disable = True

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name} ({dataset_split})")
    print(f"Output: {output_dir}")
    print(f"W&B: {wandb_project}/{wandb_run_name}")
    print("")

    # === Load Model with LoRA ===
    print("=" * 80)
    print("LOADING MODEL AND ADDING LORA ADAPTERS")
    print("=" * 80)

    # Load base model
    # Unsloth's FastVisionModel is a drop-in replacement for HF's model
    # but with optimized kernels and memory usage
    model, processor = FastVisionModel.from_pretrained(
        model_path,
        load_in_4bit=False,                    # Use full precision (more accurate)
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized checkpointing
        max_seq_length=max_seq_length,         # Context window
        dtype=torch.bfloat16,                   # bfloat16 is great for training
    )

    # Add LoRA adapters
    # LoRA (Low-Rank Adaptation) trains small adapter layers instead of the full model
    # This is WAY more efficient - we only train ~1% of parameters!
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # Keep vision encoder frozen
        finetune_language_layers=True,     # Train the language model part
        finetune_attention_modules=True,   # Add LoRA to attention
        finetune_mlp_modules=True,         # Add LoRA to MLPs

        # LoRA config
        r=lora_r,                          # Rank (32 is a good default)
        lora_alpha=lora_alpha,             # Scaling (usually 2x rank)
        lora_dropout=lora_dropout,         # Dropout (0.0 often works fine)
        bias="none",                       # Don't train bias terms
        random_state=3407,                 # For reproducibility
        target_modules="all-linear",       # Apply to all linear layers
        modules_to_save=["lm_head", "embed_tokens"],  # Also train these
    )

    # Set up chat template for the model
    # This formats inputs/outputs correctly for Gemma
    processor = get_chat_template(processor, "gemma-3")

    print(f"✓ Model loaded with LoRA adapters")
    print(f"  - Base model: {model_path}")
    print(f"  - LoRA rank: {lora_r}")
    print(f"  - Trainable params: ~1-2% of total")
    print("")

    # === Load and Prepare Dataset ===
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    # Load dataset from cache (downloaded in Stage 1)
    dataset = load_dataset(dataset_name, split=dataset_split)

    # Limit dataset size if specified (useful for testing)
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"⚠️  Limited to {len(dataset)} samples for testing")

    print(f"✓ Dataset loaded: {len(dataset)} samples")
    print("")

    # === Format Dataset ===
    # Convert dataset to chat format that Gemma expects
    # Each sample has an image and corresponding LaTeX code

    instruction = "Write the LaTeX representation for this image."

    def convert_to_conversation(sample):
        """
        Convert a dataset sample to chat format.

        Input sample has:
          - "image": PIL Image of equation
          - "text": LaTeX code for that equation

        Output format:
          - User message: instruction + image
          - Assistant message: LaTeX code
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["text"]}],
            },
        ]
        return {"messages": conversation}

    # Convert all samples
    print("Converting dataset to chat format...")
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]
    print(f"✓ Converted {len(converted_dataset)} samples")
    print("")

    # === Training Setup ===
    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    # Enable training mode (sets up gradient computation)
    FastVisionModel.for_training(model)

    # Create trainer
    # SFTTrainer is from TRL library - supervised fine-tuning trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=converted_dataset,
        processing_class=processor.tokenizer,

        # Data collator handles batching images + text
        # Unsloth's collator is optimized for vision-language models
        data_collator=UnslothVisionDataCollator(
            model=model,
            processor=processor
        ),

        # Training arguments
        args=SFTConfig(
            # === Batch size config ===
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            # Effective batch size = 4 * 4 = 16

            # === Learning rate schedule ===
            warmup_ratio=warmup_ratio,         # Warm up for 20% of training
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",        # Linear decay after warmup

            # === Logging ===
            logging_steps=logging_steps,       # Log every 10 steps
            report_to="wandb",                 # Log to W&B

            # === Checkpointing ===
            save_strategy=save_strategy,       # Save by steps
            save_steps=save_steps,             # Every 250 steps
            save_total_limit=save_total_limit, # Keep only 20 checkpoints
            output_dir=output_dir,             # Where to save

            # === Optimization ===
            gradient_checkpointing=True,       # Trade compute for memory
            gradient_checkpointing_kwargs={"use_reentrant": False},
            max_grad_norm=0.3,                 # Gradient clipping
            optim="adamw_torch_fused",         # Fastest AdamW implementation
            weight_decay=0.01,                 # L2 regularization

            # === Precision ===
            bf16=True,                         # Use bfloat16 (faster + stable)
            tf32=False,                        # Don't need TF32

            # === Vision-specific settings ===
            remove_unused_columns=False,       # Keep all columns (need images!)
            dataset_text_field="",             # We handle formatting ourselves
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,
        ),
    )

    print(f"Training config:")
    print(f"  - Effective batch size: {per_device_train_batch_size * gradient_accumulation_steps}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Epochs: {num_train_epochs}")
    print(f"  - Total steps: ~{len(converted_dataset) // (per_device_train_batch_size * gradient_accumulation_steps) * num_train_epochs}")
    print("")

    # === TRAIN! ===
    print("🚀 Starting training...")
    print("=" * 80)
    trainer_stats = trainer.train()
    print("=" * 80)
    print("✓ Training completed!")
    print("")

    # === Save Model ===
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    # Create output directories
    final_weights_dir = os.path.join(output_dir, "final_weights")  # Merged model
    final_lora_dir = os.path.join(output_dir, "final_lora")        # LoRA adapters only

    os.makedirs(final_weights_dir, exist_ok=True)
    os.makedirs(final_lora_dir, exist_ok=True)

    # Save LoRA adapters (small, ~100MB)
    print("Saving LoRA adapters...")
    model.save_pretrained(final_lora_dir)
    processor.save_pretrained(final_lora_dir)
    print(f"  ✓ LoRA adapters saved to {final_lora_dir}")

    # Optionally push LoRA to Hugging Face Hub
    if hub_id:
        print(f"Pushing LoRA to Hub: {hub_id}_lora")
        model.push_to_hub(
            f"{hub_id}_lora",
            token=os.environ["HUGGINGFACE_TOKEN"]
        )
        processor.push_to_hub(
            f"{hub_id}_lora",
            token=os.environ["HUGGINGFACE_TOKEN"]
        )
        print(f"  ✓ Pushed to {hub_id}_lora")

    # Save merged model (base + LoRA combined, ready to deploy)
    print("Saving merged model (this takes a few minutes)...")
    model.save_pretrained_merged(
        final_weights_dir,
        processor,
        save_method="merged_16bit"  # Save in 16-bit precision
    )
    print(f"  ✓ Merged model saved to {final_weights_dir}")

    # Optionally push merged model to Hub
    if hub_id:
        print(f"Pushing merged model to Hub: {hub_id}")
        model.push_to_hub_merged(
            hub_id,
            processor,
            token=os.environ["HUGGINGFACE_TOKEN"],
            save_method="merged_16bit"
        )
        print(f"  ✓ Pushed to {hub_id}")

    # CRITICAL: Commit everything to the volume
    # This persists checkpoints, final models, everything
    print("\nCommitting to volume...")
    exp_volume.commit()
    print("✓ Volume committed")

    print("")
    print("=" * 80)
    print("🎉 FINE-TUNING COMPLETE!")
    print("=" * 80)
    print(f"LoRA adapters: {final_lora_dir}")
    print(f"Merged model: {final_weights_dir}")
    if hub_id:
        print(f"Hugging Face: {hub_id} and {hub_id}_lora")
    print("")

    return {
        "status": "completed",
        "output_dir": output_dir,
        "lora_dir": final_lora_dir,
        "merged_dir": final_weights_dir,
        "hub_id": hub_id,
    }
```

Phew! That's a lot of code, but it's all there for a reason. Let me highlight the key points:

**LoRA Strategy:**
- We freeze the vision encoder (it's already good at seeing images)
- We only train LoRA adapters on the language model
- This trains ~1-2% of parameters instead of 100%
- Massively faster and more memory efficient

**Batch Size Math:**
```
Effective batch size = per_device_batch_size × gradient_accumulation_steps × num_gpus
                    = 4 × 4 × 1
                    = 16
```

**Two Save Formats:**
1. **LoRA adapters** (~100MB): Just the trained adapters. Requires base model to use.
2. **Merged model** (full size): Base model + adapters combined. Ready to deploy.

For serving, we use the merged model. For sharing or storage, LoRA adapters are more efficient.

### Running Training

**Basic run (test on small subset):**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --max-samples=100 \
  --num-train-epochs=1
```

This trains on 100 samples for 1 epoch - great for making sure everything works.

**Full training run:**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --num-train-epochs=3 \
  --learning-rate=0.0003
```

**Train and push to Hugging Face:**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --hub-id="your-username/gemma-latex-ocr" \
  --num-train-epochs=3
```

This pushes both the LoRA adapters and merged model to your HF account.

**Custom hyperparameters:**

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --lora-r=64 \
  --lora-alpha=128 \
  --per-device-train-batch-size=2 \
  --gradient-accumulation-steps=8
```

While training runs, you'll see logs streaming in real-time. And if you set up W&B, check `wandb.ai/<your-username>/GemmaFinetuning` to see beautiful charts of loss curves, learning rate schedules, GPU utilization, everything.

## Stage 4: Export and Merge Model (Optional)

Okay, so after training, you have LoRA adapters saved. The training function already saves both LoRA adapters AND the merged model. But let's say you only saved LoRA adapters (to save space), and now you want to create a standalone merged model. That's what this stage is for.

```python
@app.function(
    image=FINETUNING_GPU_IMAGE,
    volumes=VOLUME_CONFIG,
    gpu=TRAINING_GPU_CONFIG,              # Need same GPU as training
    secrets=[huggingface_secret, Secret.from_dotenv()],
    timeout=2 * HOURS,                    # Merging takes ~10-30 minutes
)
def export_model(
    lora_model_path: str = f"{OUTPUT_DIR_DEFAULT}",  # Where LoRA adapters are
    output_path: str = None,                          # Where to save merged model
    hub_model_id: str = None,                         # Optional: push to HF Hub
    push_to_hub: bool = True,                         # Whether to push
):
    """
    Export LoRA adapters and merge them with base model.

    Why? Two reasons:
    1. Merged models are easier to deploy (no need to load base + adapters separately)
    2. Merged models can be quantized for faster inference
    """
    from unsloth import FastVisionModel
    import os

    # Set HF token for pushing to Hub
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    print("=" * 80)
    print("LOADING LORA MODEL AND MERGING")
    print("=" * 80)
    print(f"LoRA path: {lora_model_path}")

    # Load the LoRA model
    # This loads base model + LoRA adapters
    model, processor = FastVisionModel.from_pretrained(
        lora_model_path,           # Path to LoRA adapters
        load_in_4bit=False,        # Load in full precision
    )

    # Prepare for inference
    # This merges the LoRA weights into the base model
    FastVisionModel.for_inference(model)

    print("✓ Model loaded and LoRA weights merged")
    print("")

    # === Save or Push ===
    if push_to_hub and hub_model_id:
        # Push merged model to Hugging Face Hub
        print(f"Pushing merged model to Hub: {hub_model_id}")
        model.push_to_hub_merged(
            hub_model_id,
            processor,
            token=os.environ["HUGGINGFACE_TOKEN"],
            save_method="merged_16bit",  # Save in 16-bit (good balance)
        )
        print(f"✓ Pushed to https://huggingface.co/{hub_model_id}")
    else:
        # Save locally to volume
        if output_path is None:
            output_path = f"{lora_model_path}_merged"

        print(f"Saving merged model to: {output_path}")
        model.save_pretrained_merged(
            output_path,
            processor,
            save_method="merged_16bit"
        )
        print(f"✓ Saved to {output_path}")

    # Commit to volume
    exp_volume.commit()

    print("")
    print("=" * 80)
    print("✓ EXPORT COMPLETE!")
    print("=" * 80)

    return {
        "status": "completed",
        "lora_path": lora_model_path,
        "merged_path": output_path if not push_to_hub else hub_model_id,
    }
```

**When to use this:**
- You only saved LoRA adapters during training (to save disk space)
- You want to create a standalone model for deployment
- You want to push to HuggingFace Hub after training

**Run it:**

```bash
# Export and save to volume
modal run FinetuneGemmaUnslothModal.py::export_model \
  --lora-model-path="/data/Finetuned_Gemma_3_4b_it/final_lora"

# Export and push to HuggingFace
modal run FinetuneGemmaUnslothModal.py::export_model \
  --lora-model-path="/data/Finetuned_Gemma_3_4b_it/final_lora" \
  --hub-model-id="your-username/gemma-latex-merged" \
  --push-to-hub=True
```

## Stage 5: Serving with vLLM

Alright, now let's deploy our model for real-time inference. We're using vLLM, which is basically the industry standard for serving LLMs at scale.

**Why vLLM?**
- **Fast**: Optimized attention kernels, continuous batching
- **Scalable**: Handles thousands of requests per second
- **Compatible**: OpenAI-compatible API (drop-in replacement)
- **Auto-scaling**: Modal handles spinning up/down instances based on load

### vLLM Image (Separate from Training)

We use a different image for serving because vLLM has different dependencies than training.

```python
VLLM_CUDA_VERSION = "12.8.1"
VLLM_CUDA_TAG = f"{VLLM_CUDA_VERSION}-devel-ubuntu24.04"

VLLM_GPU_IMAGE = (
    # Start with CUDA base
    ModalImage.from_registry(f"nvidia/cuda:{VLLM_CUDA_TAG}", add_python="3.12")

    # Install system dependencies for vLLM
    .apt_install("libopenmpi-dev", "libnuma-dev")  # For distributed inference

    # Upgrade pip and install uv
    .run_commands("pip install --upgrade pip")
    .run_commands("pip install uv")

    # Install vLLM (latest version)
    .run_commands("uv pip install vllm -U --system")

    # Install supporting packages
    .pip_install(
        "datasets",                       # For eval/testing
        "pillow",                         # Image handling
        "huggingface_hub[hf_transfer]",  # Fast model downloads
        "requests",                       # HTTP requests
        "numpy",                          # Numerical ops
    )

    # Install flash-attention (required for vLLM)
    # Must be installed separately with --no-build-isolation
    .run_commands(
        "uv pip install 'flash-attn>=2.7.1,<=2.8.0' --no-build-isolation --system"
    )

    # Enable fast HF downloads
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```

**Why separate image?**
- vLLM and training have overlapping dependencies that can conflict
- vLLM image is lighter (no training frameworks)
- Faster to build and deploy

### Serving Configuration

```python
# Which model to serve (path on volume)
DEFAULT_SERVE_MODEL = "/data/Finetuned_Gemma_3_4b_it/final_weights"

# GPU for serving (can be different from training!)
SERVE_GPU = "L40S"        # L40S is great for inference (~$1/hr)
SERVE_NUM_GPUS = 1
SERVE_GPU_CONFIG = f"{SERVE_GPU}:{SERVE_NUM_GPUS}"

VLLM_PORT = 8000          # Internal port
```

**GPU choice for serving:**
- **L40S**: Best price/performance for inference ($1/hr)
- **A100-40GB**: If you need higher throughput ($2.50/hr)
- **A100-80GB**: For very large models or high batch sizes ($3.50/hr)

### The Serve Function

```python
@app.function(
    image=VLLM_GPU_IMAGE,
    gpu=SERVE_GPU_CONFIG,                # L40S for serving
    scaledown_window=3 * 60,             # Scale to 0 after 3 min idle (saves $$$)
    secrets=[huggingface_secret],        # Need HF token
    volumes=VOLUME_CONFIG,                # Mount our volume (has the model)
    max_containers=2,                     # Auto-scale up to 2 instances
    timeout=24 * HOURS,
)
@modal.concurrent(max_inputs=50)         # Handle 50 concurrent requests per instance
@modal.web_server(port=8000, startup_timeout=5 * 60)  # Expose as web server
def serve_vllm():
    """
    Serve the fine-tuned model using vLLM.

    This creates an OpenAI-compatible API endpoint that:
    - Auto-scales from 0 to max_containers based on load
    - Shuts down after 3 minutes of inactivity (cost optimization!)
    - Handles up to 50 concurrent requests per container
    """
    import subprocess
    import os

    # Set HF token (might need to download model files)
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    print("=" * 80)
    print("STARTING VLLM SERVER")
    print("=" * 80)
    print(f"Model: {DEFAULT_SERVE_MODEL}")
    print(f"Port: {VLLM_PORT}")
    print(f"GPU: {SERVE_GPU_CONFIG}")
    print("")

    # Build vLLM command
    cmd = [
        "vllm", "serve",                           # vLLM serve command
        "--uvicorn-log-level=info",                # Logging level
        DEFAULT_SERVE_MODEL,                       # Path to model
        "--host", "0.0.0.0",                       # Listen on all interfaces
        "--port", str(VLLM_PORT),                  # Port to serve on
        "--enforce-eager",                         # Faster startup (skip torch.compile)
        "--tensor-parallel-size", str(SERVE_NUM_GPUS),  # How many GPUs to use
        "--gpu-memory-utilization", "0.4",         # Use 40% of GPU memory (be conservative)
        "--trust-remote-code",                     # Allow custom model code
    ]

    print(f"Command: {' '.join(cmd)}")
    print("")
    print("🚀 Starting vLLM server...")
    print("=" * 80)

    # Start vLLM in background
    # Popen returns immediately, server keeps running
    subprocess.Popen(" ".join(cmd), shell=True)
```

**Key configuration options:**

1. **`scaledown_window=3*60`**: This is HUGE for cost savings. If there are no requests for 3 minutes, Modal shuts down the container. You pay $0 when idle!

2. **`max_containers=2`**: Modal will automatically spin up a second instance if the first one gets too many requests. Load balancing happens automatically.

3. **`@modal.concurrent(max_inputs=50)`**: Each instance can handle 50 concurrent requests. If you get more than 50, Modal queues them or spins up instance #2.

4. **`gpu-memory-utilization=0.4`**: Use only 40% of GPU memory. vLLM is memory-efficient, and this leaves headroom for request spikes.

### Deploying the Server

To deploy and keep it running:

```bash
modal deploy FinetuneGemmaUnslothModal.py
```

This creates a persistent deployment that stays alive (but auto-scales to 0 when idle).

**Get the URL:**

After deploying, Modal prints the URL. Or find it with:

```bash
modal app list
```

You'll get something like: `https://your-username--finetuned-gemma-3-4b-it-serve-vllm.modal.run`

### Using the API

The server exposes an OpenAI-compatible API. Here's how to use it:

```python
from openai import OpenAI
import base64

# Create client pointing to your Modal endpoint
client = OpenAI(
    base_url="https://your-endpoint.modal.run/v1",  # Your Modal URL + /v1
    api_key="EMPTY"  # Modal doesn't require API key (it's behind Modal auth)
)

# Encode image to base64
with open("equation.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Make request (just like OpenAI!)
response = client.chat.completions.create(
    model="/data/Finetuned_Gemma_3_4b_it/final_weights",  # Model path
    messages=[
        {
            "role": "user",
            "content": [
                # Send image as base64
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                },
                # Send text prompt
                {
                    "type": "text",
                    "text": "Write the LaTeX representation for this image."
                },
            ],
        },
    ],
    temperature=0.1,      # Low temp for deterministic output
    max_tokens=512,       # Max length of response
)

# Print the LaTeX code
print(response.choices[0].message.content)
```

**Example output:**
```
\frac{d}{dx} \left( x^2 + 2x + 1 \right) = 2x + 2
```

### Testing the Deployment

Quick test script:

```python
import requests
import base64

# Your Modal endpoint
url = "https://your-endpoint.modal.run/v1/chat/completions"

# Load and encode image
with open("test_equation.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Make request
response = requests.post(
    url,
    json={
        "model": "/data/Finetuned_Gemma_3_4b_it/final_weights",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": "Write the LaTeX representation for this image."}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

**Pro tip:** The first request after the server scales from 0 will take 30-60 seconds (model loading). Subsequent requests are instant.

## Stage 6: Evaluation

Alright, let's measure how good our model actually is. We'll use real metrics: exact match accuracy, character error rate, and word error rate.

### Evaluation Image (Lightweight, CPU-only)

```python
EVAL_IMAGE = (
    # Lightweight Debian base (no CUDA needed for eval)
    ModalImage.debian_slim(python_version="3.12")

    # Install evaluation dependencies
    .pip_install(
        "openai",                         # To call our vLLM endpoint
        "datasets",                       # Load test dataset
        "pillow",                         # Image processing
        "numpy",                          # Numerical ops
        "jiwer",                          # Word/Character Error Rate metrics
        "nltk",                           # NLP utilities
        "tqdm",                           # Progress bars
        "huggingface_hub[hf_transfer]",  # Fast dataset downloads
    )

    # Enable fast downloads
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
```

**Why CPU for evaluation?**
- Evaluation just calls our API endpoint (which has the GPU)
- Processing responses doesn't need GPU
- Saves money!

### The Evaluation Function

```python
@app.function(
    image=EVAL_IMAGE,                # Lightweight CPU image
    volumes=VOLUME_CONFIG,            # Access cached datasets
    secrets=[huggingface_secret],    # HF token for datasets
    timeout=2 * HOURS,                # Eval can take a while
    # No GPU! Runs on CPU
)
def evaluate_model(
    endpoint_url: str = None,                                  # vLLM endpoint (auto-detected)
    model_name: str = "/data/Finetuned_Gemma_3_4b_it/final_weights",
    dataset_name: str = "unsloth/LaTeX_OCR",                  # Test dataset
    dataset_split: str = "test",                              # Use test split
    max_samples: int = 100,                                    # How many to evaluate
    max_parallel_requests: int = 8,                           # Concurrent requests
    temperature: float = 0.1,                                 # Low temp for consistency
    max_tokens: int = 512,                                    # Max response length
):
    """
    Evaluate the fine-tuned model on LaTeX OCR test set.

    Metrics:
    - Exact Match Accuracy: % of perfect predictions
    - Character Error Rate (CER): Edit distance at character level
    - Word Error Rate (WER): Edit distance at word level
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from openai import OpenAI
    from datasets import load_dataset
    from jiwer import wer, cer
    from tqdm import tqdm
    import base64
    from io import BytesIO
    import os

    print("=" * 80)
    print("EVALUATING MODEL")
    print("=" * 80)

    # === Get endpoint URL ===
    if endpoint_url is None:
        # Auto-retrieve the vLLM endpoint URL
        print("Auto-detecting vLLM endpoint...")
        endpoint_url = serve_vllm.get_web_url().rstrip("/") + "/v1"

    print(f"Endpoint: {endpoint_url}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name} ({dataset_split})")
    print(f"Max samples: {max_samples}")
    print("")

    # === Load test dataset ===
    print("Loading test dataset...")
    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    dataset = load_dataset(dataset_name, split=dataset_split)

    # Limit to max_samples
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"✓ Loaded {len(dataset)} samples")
    print("")

    # === Set up OpenAI client ===
    client = OpenAI(
        base_url=endpoint_url,
        api_key="EMPTY"  # Modal doesn't require API key
    )

    # === Helper function to encode images ===
    def encode_image_to_base64(image):
        """Convert PIL Image to base64 string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode()

    # === Run inference on all samples (in parallel) ===
    def run_inference(sample, idx):
        """
        Run inference on a single sample.

        Returns:
            dict with "prediction" and "ground_truth"
        """
        try:
            # Encode image
            image_base64 = encode_image_to_base64(sample["image"])

            # Call API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                            },
                            {
                                "type": "text",
                                "text": "Write the LaTeX representation for this image."
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Extract prediction
            prediction = response.choices[0].message.content.strip()
            ground_truth = sample["text"].strip()

            return {
                "prediction": prediction,
                "ground_truth": ground_truth,
            }
        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            return {
                "prediction": "",
                "ground_truth": sample["text"].strip(),
            }

    # Run evaluation with parallel requests
    print(f"Running inference on {len(dataset)} samples...")
    print(f"Parallelism: {max_parallel_requests} concurrent requests")
    print("")

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        # Submit all tasks
        futures = [
            executor.submit(run_inference, dataset[i], i)
            for i in range(len(dataset))
        ]

        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(dataset), desc="Evaluating"):
            results.append(future.result())

    # === Calculate metrics ===
    print("")
    print("=" * 80)
    print("CALCULATING METRICS")
    print("=" * 80)

    predictions = [r["prediction"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]

    # Exact match accuracy
    exact_matches = sum(p == g for p, g in zip(predictions, ground_truths))
    exact_match_accuracy = exact_matches / len(results)

    # Character Error Rate (CER)
    # Lower is better, 0 = perfect
    character_error_rate = cer(ground_truths, predictions)

    # Word Error Rate (WER)
    # Lower is better, 0 = perfect
    word_error_rate = wer(ground_truths, predictions)

    # === Print results ===
    print("")
    print("📊 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Samples evaluated: {len(results)}")
    print(f"")
    print(f"Exact Match Accuracy:  {exact_match_accuracy:.2%}  ({exact_matches}/{len(results)})")
    print(f"Character Error Rate:  {character_error_rate:.2%}  (lower is better)")
    print(f"Word Error Rate:       {word_error_rate:.2%}  (lower is better)")
    print("=" * 80)

    # === Print example predictions ===
    print("")
    print("📝 EXAMPLE PREDICTIONS (first 5)")
    print("=" * 80)
    for i in range(min(5, len(results))):
        print(f"\nSample {i+1}:")
        print(f"  Ground Truth: {results[i]['ground_truth']}")
        print(f"  Prediction:   {results[i]['prediction']}")
        print(f"  Match: {'✓' if results[i]['prediction'] == results[i]['ground_truth'] else '✗'}")
    print("=" * 80)

    # Save full results to volume
    results_file = f"/data/Finetuned_Gemma_3_4b_it/eval_results_{dataset_split}.json"
    import json
    with open(results_file, "w") as f:
        json.dump({
            "metrics": {
                "exact_match_accuracy": exact_match_accuracy,
                "character_error_rate": character_error_rate,
                "word_error_rate": word_error_rate,
            },
            "num_samples": len(results),
            "examples": results[:20],  # Save first 20 examples
        }, f, indent=2)

    exp_volume.commit()
    print(f"\n✓ Full results saved to {results_file}")

    return {
        "status": "completed",
        "metrics": {
            "exact_match_accuracy": exact_match_accuracy,
            "character_error_rate": character_error_rate,
            "word_error_rate": word_error_rate,
        },
        "num_samples": len(results),
        "examples": results[:10],  # Return first 10 examples
    }
```

**What the metrics mean:**

1. **Exact Match Accuracy**: The gold standard. Did we get it 100% right? For LaTeX, even a missing space matters.

2. **Character Error Rate (CER)**: How many character edits (insert/delete/replace) to go from prediction to ground truth? Lower is better. 0% = perfect, 100% = complete garbage.

3. **Word Error Rate (WER)**: Same as CER but at word level. More forgiving for LaTeX because `\frac{a}{b}` has multiple "words".

### Running Evaluation

**Basic run:**

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model
```

This auto-detects your deployed vLLM endpoint and evaluates on 100 samples.

**Evaluate more samples:**

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model \
  --max-samples=500 \
  --max-parallel-requests=16
```

**Custom endpoint:**

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model \
  --endpoint-url="https://your-custom-endpoint.modal.run/v1" \
  --max-samples=1000
```

**Example output:**

```
📊 EVALUATION RESULTS
================================================================================
Samples evaluated: 100

Exact Match Accuracy:  78.00%  (78/100)
Character Error Rate:  5.23%  (lower is better)
Word Error Rate:       8.45%  (lower is better)
================================================================================

📝 EXAMPLE PREDICTIONS (first 5)
================================================================================

Sample 1:
  Ground Truth: \frac{d}{dx} \left( x^2 + 2x + 1 \right) = 2x + 2
  Prediction:   \frac{d}{dx} \left( x^2 + 2x + 1 \right) = 2x + 2
  Match: ✓

Sample 2:
  Ground Truth: \int_{0}^{1} x^2 dx = \frac{1}{3}
  Prediction:   \int_{0}^{1} x^2 dx = \frac{1}{3}
  Match: ✓

...
```

## Complete Workflow Example

Let me show you how I'd actually use this end-to-end:

### 1. Download Everything (One Time)

```bash
# Download dataset (CPU, cheap)
modal run FinetuneGemmaUnslothModal.py::download_datasets

# Download model (L40S, ~$1 for 10 minutes)
modal run FinetuneGemmaUnslothModal.py::download_models
```

**Cost so far:** ~$1
**Time:** ~15 minutes

### 2. Quick Test Run (Make Sure It Works)

```bash
# Train on 100 samples for 1 epoch
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --max-samples=100 \
  --num-train-epochs=1 \
  --save-steps=50
```

**Cost:** ~$3-5 (A100-80GB for 30-60 minutes)
**Time:** 30-60 minutes

If this works, you know your pipeline is solid.

### 3. Full Training Run

```bash
# Production training with HF Hub push
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --hub-id="your-username/gemma-latex-ocr" \
  --num-train-epochs=3 \
  --learning-rate=0.0003 \
  --per-device-train-batch-size=4 \
  --gradient-accumulation-steps=4
```

**Cost:** ~$20-40 (A100-80GB for 4-8 hours depending on dataset size)
**Time:** 4-8 hours

While this runs, go touch grass. Check W&B dashboard occasionally to make sure loss is going down.

### 4. Deploy for Serving

```bash
modal deploy FinetuneGemmaUnslothModal.py
```

**Cost:** $0 when idle, ~$1/hr when active (L40S)

Modal gives you a URL. Save it.

### 5. Evaluate

```bash
modal run FinetuneGemmaUnslothModal.py::evaluate_model \
  --max-samples=500
```

**Cost:** ~$0.10 (CPU for 10-20 minutes)
**Time:** 10-20 minutes

Check your metrics. If accuracy is good (>75%), you're golden. If not, tweak hyperparameters and go back to step 3.

### 6. Use in Production

```python
# In your application
from openai import OpenAI

client = OpenAI(
    base_url="https://your-endpoint.modal.run/v1",
    api_key="EMPTY"
)

# Your app can now read LaTeX from images!
```

**Total cost for full pipeline:** ~$25-50
**Time:** 1 day (mostly waiting for training)

Compare this to managing your own GPU infrastructure... yeah, Modal wins.

## Hyperparameter Tuning Tips

### For Better Accuracy

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --lora-r=64 \              # Higher rank = more capacity
  --lora-alpha=128 \         # Scale accordingly
  --learning-rate=0.0001 \   # Lower LR = more stable
  --num-train-epochs=5       # More epochs
```

**Trade-off:** Slower training, higher cost, but better results.

### For Faster Iteration

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --lora-r=16 \              # Lower rank = faster
  --lora-alpha=32 \
  --learning-rate=0.0005 \   # Higher LR = faster convergence
  --num-train-epochs=2 \
  --max-samples=5000         # Smaller dataset
```

**Trade-off:** Lower accuracy, but 2-3x faster training.

### For Memory Issues

If you get OOM errors:

```bash
modal run FinetuneGemmaUnslothModal.py::fine_tune_unsloth \
  --per-device-train-batch-size=2 \     # Smaller batches
  --gradient-accumulation-steps=8 \     # Maintain effective batch size
  --max-seq-length=4096                  # Shorter sequences
```

Or switch to A100-80GB if you're on A100-40GB.

## Common Issues and Solutions

### "Secret not found"

**Error:** `Modal Secret "secrets-hf-wandb" not found`

**Fix:**
```bash
modal secret create secrets-hf-wandb \
  HUGGINGFACE_TOKEN=hf_xxx \
  WANDB_API_KEY=xxx
```

### CUDA Out of Memory

**Error:** `CUDA out of memory`

**Fixes:**
1. Reduce batch size: `--per-device-train-batch-size=2`
2. Reduce sequence length: `--max-seq-length=4096`
3. Use smaller LoRA rank: `--lora-r=16 --lora-alpha=32`
4. Switch to A100-80GB

### Image Build Timeout

**Error:** Image build exceeds timeout

**Fix:** First build takes 15-20 minutes. This is normal. Modal caches it. Grab a coffee.

### vLLM Server Not Responding

**Error:** `Could not connect to endpoint`

**Fix:**
```bash
# Make sure it's deployed
modal app list

# If not running, deploy it
modal deploy FinetuneGemmaUnslothModal.py
```

The first request after deploy takes 30-60 seconds (cold start). Be patient.

### Evaluation Fails

**Error:** Various errors during eval

**Checks:**
1. Is vLLM running? `modal app list`
2. Is the endpoint URL correct?
3. Is the model path correct in the eval function?

## Cost Breakdown

Based on Modal pricing (approximate):

### Training
- **Download dataset:** $0.001 (CPU, 5 min)
- **Download model:** $1 (L40S, 10 min)
- **Test training:** $5 (A100-80GB, 1 hour)
- **Full training:** $25-40 (A100-80GB, 6-10 hours)

### Serving (pay per use)
- **Idle:** $0/month (auto-scales to 0)
- **Active:** ~$1/hour (L40S)
- **Typical monthly cost:** $5-20 (depends on usage)

### Evaluation
- **CPU cost:** ~$0.10 per eval run

### Storage
- **Volumes:** Free up to 50GB
- **This project:** ~15GB = $0/month

**Total for complete pipeline:** $30-50 one-time + $5-20/month for serving

## What's Next?

You've built a complete production ML pipeline. Here's what you can do next:

1. **Try different models:** Replace Gemma with Llama, Qwen, or any other vision-language model. Just change `BASE_MODEL_NAME`.

2. **Use your own dataset:** Got images + text pairs? Upload to HuggingFace, point the script at it.

3. **Optimize serving:** Experiment with different GPUs, batch sizes, quantization.

4. **Add more metrics:** BLEU score, semantic similarity, whatever matters for your use case.

5. **Build an app:** You have an API. Now build a web app that uses it!

## Resources

- **[Original Unsloth Colab](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B)-Vision.ipynb)** - Where this all started
- **[Unsloth Documentation](https://docs.unsloth.ai/)** - Deep dive into Unsloth
- **[Modal Documentation](https://modal.com/docs)** - Everything about Modal
- **[vLLM Documentation](https://docs.vllm.ai/)** - Serving optimization
- **[Gemma Model Card](https://huggingface.co/google/gemma-3-4b-it)** - About the base model
- **[LoRA Paper](https://arxiv.org/abs/2106.09685)** - The theory behind it

---

## Wrapping Up

You just built what most companies would consider their "production ML infrastructure":
- Dataset management
- Distributed training
- Model versioning
- API deployment
- Evaluation pipelines

All in one Python file, running on Modal. No Kubernetes, no Docker nightmares, no infrastructure headaches.

The Unsloth Colab notebook showed you how to train on a single GPU. This tutorial showed you how to take that exact workflow and productionize it - separate stages, proper caching, auto-scaling deployment, real evaluation metrics.

This is how I actually do ML work nowadays. Write code locally, run on Modal's GPUs, deploy with one command.

Got questions? Hit me up on Twitter [@adithya_s_k](https://x.com/adithya_s_k)!

Now go build something cool with this. 🚀