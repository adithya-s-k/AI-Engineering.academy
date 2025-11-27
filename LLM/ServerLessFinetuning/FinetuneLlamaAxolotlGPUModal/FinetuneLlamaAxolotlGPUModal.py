from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Time constants
HOURS = 60 * 60
# GPU Configuration
GPU_TYPE = "a100-80gb"  # Default GPU type (can be: a100-80gb, a100-40gb, l40s, etc.)
# Training Configuration
WANDB_PROJECT_DEFAULT = "Llama-70b-MultiGPU-finetune"

# =============================================================================
# MODAL APP VOLUME AND SECRET SETUP
# =============================================================================

app = App("Finetuned_Llama_70b_Axolotl_MultiGPU")
# Create volumes for persistent storage
exp_volume = Volume.from_name("Finetuned_Llama_70b_Axolotl", create_if_missing=True)
# Configure volume mounting points
VOLUME_CONFIG = {
    "/data": exp_volume,
}
huggingface_secret = Secret.from_name("secrets-hf-wandb")


# =============================================================================
# MODEL IMAGE SETUP
# =============================================================================


# This is the original Axolotl image, it can be used but it opens JupyterLab by default
# AXOLOTL_IMAGE = ModalImage.from_registry(
#     "axolotlai/axolotl-cloud:main-latest", add_python="3.12"
# ).env(
#     {
#         "JUPYTER_ENABLE_LAB": "no",  # Disable JupyterLab auto-start
#         "JUPYTER_TOKEN": "",  # Disable Jupyter token requirement
#         "HF_HOME": "/data/.cache",  # Set HF cache root under /data
#     }
# )

# Custom CUDA image with Axolotl and dependencies pre-installed
CUDA_VERSION = "12.8.1"
CUDA_FLAVOR = "devel"
CUDA_OS = "ubuntu24.04"
CUDA_TAG = f"{CUDA_VERSION}-{CUDA_FLAVOR}-{CUDA_OS}"

# Define the GPU image for fine-tuning with Unsloth
AXOLOTL_IMAGE = (
    ModalImage.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
    )
    .uv_pip_install(
        [
            "torch",
            "torchvision",
            "torchaudio",  # optional but often bundled with torch
        ]
    )
    .run_commands(
        "uv pip install --no-deps -U packaging setuptools wheel ninja --system"
    )
    .run_commands("uv pip install --no-build-isolation axolotl[deepspeed] --system")
    .run_commands(
        "UV_NO_BUILD_ISOLATION=1 uv pip install flash-attn --no-build-isolation --system"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": "/data/.cache",  # Set HF cache root under /data
        }
    )
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def write_config_to_volume(
    train_config_yaml: str,
    config_path: str = "/data/config.yml",
    update_paths: bool = True,
) -> dict:
    """Write YAML configuration to volume with optional path updates."""
    import os
    import yaml

    config_dict = yaml.safe_load(train_config_yaml)

    if update_paths and "output_dir" in config_dict:
        config_dict["output_dir"] = config_dict["output_dir"].replace(
            "./outputs", "/data/outputs"
        )

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    exp_volume.commit()
    return config_dict


# =============================================================================
# TRAINING CONFIGURATION
# You can fine more Configuration options here:
# https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples
# =============================================================================

TRAIN_CONFIG_YAML = f"""
base_model: NousResearch/Meta-Llama-3-8B-Instruct
# optionally might have model_type or tokenizer_type
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer
# Automatically upload checkpoint and final model to HF
# hub_model_id: username/custom_model_name

load_in_8bit: true
load_in_4bit: false

chat_template: llama3
datasets:
  - path: fozziethebeat/alpaca_messages_2k_test
    type: chat_template
dataset_prepared_path: /data/prepared_datasets/alpaca_2k
val_set_size: 0.05
output_dir: /data/outputs/lora-out

sequence_len: 4096
sample_packing: false

adapter: lora
lora_model_dir:
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

wandb_project: {WANDB_PROJECT_DEFAULT}
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:

gradient_accumulation_steps: 4
micro_batch_size: 8
num_epochs: 4
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.0002

bf16: auto
tf32: false

gradient_checkpointing: true
resume_from_checkpoint:
logging_steps: 1
flash_attention: true

warmup_ratio: 0.1
evals_per_epoch: 4
saves_per_epoch: 4
weight_decay: 0.0
special_tokens:
   pad_token: <|end_of_text|>
"""

# =============================================================================
# PREPROCESSING FUNCTION
# =============================================================================

# GPU Configuration for preprocessing (single GPU)
PREPROCESS_NUM_GPUS = 1
PREPROCESS_GPU_CONFIG = f"{GPU_TYPE}:{PREPROCESS_NUM_GPUS}"


@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=24 * HOURS,
    gpu=PREPROCESS_GPU_CONFIG,
)
def process_datasets(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
):
    """Preprocess and tokenize dataset before training using Axolotl."""
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    config_dict = write_config_to_volume(train_config_yaml, config_path, True)
    exp_volume.commit()

    print("Starting dataset preprocessing...")
    try:
        subprocess.run(["axolotl", "preprocess", config_path], check=True)
        print("✓ Preprocessing completed")
        exp_volume.commit()

        return {
            "status": "completed",
            "config_path": config_path,
            "preprocessed_data_path": config_dict.get("dataset_prepared_path"),
            "output_dir": config_dict.get("output_dir"),
        }
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Preprocessing failed: {e}")


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

# GPU Configuration for training (2-8 GPUs for multi-GPU training)
TRAIN_NUM_GPUS = 4  # Can be adjusted from 2 to 8
TRAIN_GPU_CONFIG = f"{GPU_TYPE}:{TRAIN_NUM_GPUS}"


@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=24 * HOURS,
    gpu=TRAIN_GPU_CONFIG,
)
def train_model(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
):
    """
    Train or fine-tune a model using Axolotl with multi-GPU support.
    All configuration is defined in the YAML file.
    Uses accelerate for multi-GPU training.

    Args:
        train_config_yaml: YAML configuration content as string
        config_path: Path where config will be written on the volume

    Returns:
        dict: Contains training status and output paths
    """
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]
    os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_DEFAULT

    # Write config to volume using global helper function
    config_dict = write_config_to_volume(
        train_config_yaml=train_config_yaml,
        config_path=config_path,
        update_paths=True,
    )

    exp_volume.commit()

    # Run Axolotl training with accelerate for multi-GPU support
    print(f"Starting training with {TRAIN_NUM_GPUS} GPUs...")
    cmd = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(TRAIN_NUM_GPUS),
        "--num_machines",
        "1",
        "--mixed_precision",
        "bf16",
        "--dynamo_backend",
        "no",
        "-m",
        "axolotl.cli.train",
        config_path,
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✓ Training completed")

        # Commit trained model to volume
        exp_volume.commit()

        return {
            "status": "completed",
            "config_path": config_path,
            "output_dir": config_dict.get("output_dir"),
            "base_model": config_dict.get("base_model"),
            "num_gpus": TRAIN_NUM_GPUS,
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Training failed: {e}")


# =============================================================================
# MERGE LORA FUNCTION
# =============================================================================

# GPU Configuration for merging LoRA (single GPU)
MERGE_NUM_GPUS = 1
MERGE_GPU_CONFIG = f"{GPU_TYPE}:{MERGE_NUM_GPUS}"


@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=4 * HOURS,
    gpu=MERGE_GPU_CONFIG,
)
def merge_lora(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
    lora_model_dir: str = None,
):
    """
    Merge trained LoRA adapters into the base model.

    Args:
        train_config_yaml: YAML configuration content as string
        config_path: Path where config will be written on the volume
        lora_model_dir: Path to LoRA adapter directory (optional, uses config if not provided)

    Returns:
        dict: Contains merge status and output paths
    """
    import os
    import subprocess

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Write config to volume
    config_dict = write_config_to_volume(
        train_config_yaml=train_config_yaml,
        config_path=config_path,
        update_paths=True,
    )

    exp_volume.commit()

    # Build merge command
    print("Starting LoRA merge...")
    cmd = ["axolotl", "merge-lora", config_path]

    if lora_model_dir:
        cmd.extend(["--lora-model-dir", lora_model_dir])

    try:
        subprocess.run(cmd, check=True)
        print("✓ LoRA merge completed")

        # Commit merged model to volume
        exp_volume.commit()

        return {
            "status": "completed",
            "config_path": config_path,
            "output_dir": config_dict.get("output_dir"),
            "lora_model_dir": lora_model_dir or config_dict.get("lora_model_dir"),
        }

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LoRA merge failed: {e}")


# =============================================================================
# INFERENCE FUNCTION
# =============================================================================

# GPU Configuration for inference (single GPU)
INFERENCE_NUM_GPUS = 1
INFERENCE_GPU_CONFIG = f"{GPU_TYPE}:{INFERENCE_NUM_GPUS}"


@app.function(
    image=AXOLOTL_IMAGE,
    volumes=VOLUME_CONFIG,
    secrets=[huggingface_secret],
    timeout=1 * HOURS,
    gpu=INFERENCE_GPU_CONFIG,
)
def run_inference(
    train_config_yaml: str = TRAIN_CONFIG_YAML,
    config_path: str = "/data/config.yml",
    prompt: str = "Hello, how are you?",
    lora_model_dir: str = None,
    base_model: str = None,
):
    """
    Run inference using the trained model.

    Args:
        train_config_yaml: YAML configuration content as string
        config_path: Path where config will be written on the volume
        prompt: Input prompt for inference
        lora_model_dir: Path to LoRA adapter directory (optional)
        base_model: Path to base or merged model (optional)

    Returns:
        dict: Contains inference output and metadata
    """
    import os
    import subprocess
    import tempfile

    os.environ["HF_TOKEN"] = os.environ["HUGGINGFACE_TOKEN"]

    # Write config to volume
    config_dict = write_config_to_volume(
        train_config_yaml=train_config_yaml,
        config_path=config_path,
        update_paths=True,
    )

    # Build inference command
    print("Starting inference...")
    print(f"Prompt: {prompt}")
    print("-" * 80)

    cmd = ["axolotl", "inference", config_path]

    if lora_model_dir:
        cmd.extend(["--lora-model-dir", lora_model_dir])
    if base_model:
        cmd.extend(["--base-model", base_model])

    # Write prompt to temp file and pipe it
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write(prompt)
            prompt_file = f.name

        # Run inference with prompt piped from file
        with open(prompt_file, "r") as f:
            result = subprocess.run(
                cmd,
                stdin=f,
                capture_output=True,
                text=True,
                check=True,
            )

        print("✓ Inference completed")
        print("\n" + "=" * 80)
        print("MODEL OUTPUT:")
        print("=" * 80)
        print(result.stdout)
        print("=" * 80)

        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)

        response_dict = {
            "status": "completed",
            "prompt": prompt,
            "output": result.stdout,
            "model": base_model or config_dict.get("base_model"),
        }

        return response_dict

    except subprocess.CalledProcessError as e:
        print(f"Error output: {e.stderr}")
        print(f"Command output: {e.stdout}")
        raise RuntimeError(f"Inference failed: {e}")
    finally:
        # Clean up temp file
        import os as os_module

        if "prompt_file" in locals():
            os_module.unlink(prompt_file)
