"""
Train nanochat on Modal: Build Your Own ChatGPT from Scratch

Complete nanochat pipeline - from tokenizer training to a functional ChatGPT clone.

Pipeline stages:
1. Tokenizer Training - Custom BPE tokenizer (65K vocab)
2. Base Model Pretraining - GPT on FineWeb dataset
3. Midtraining - Conversation tokens and tool use
4. Supervised Fine-tuning - Task-specific training
5. Reinforcement Learning - Optional RL on GSM8K
6. Comprehensive Evaluation - CORE, ARC, GSM8K, HumanEval, MMLU
7. Inference - Chat CLI and Web UI

GPU Requirements:
- Recommended: 4-8x A100 80GB (~4 hours, ~$96)
- Minimum: 1x A100 80GB (8x longer)
- Testing: 1x A100 40GB (reduced batch sizes)

Setup:
    1. Clone nanochat: git clone https://github.com/karpathy/nanochat.git

    2. Optional - Set up secrets for WandB:
       modal secret create nanochat-secrets WANDB_API_KEY=your_key HUGGINGFACE_TOKEN=your_token

Usage:
    modal run TrainNanochatModal.py  # Full pipeline
    modal run TrainNanochatModal.py::main --num-data-shards=8 --depth=12  # Quick test
    modal run TrainNanochatModal.py::chat_cli --source=sft --prompt="Why is the sky blue?"
    modal run TrainNanochatModal.py::chat_web --source=sft  # Web UI
"""

from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

MINUTES = 60
HOURS = 60 * 60

GPU_TYPE = "a100-80gb"

# Multi-GPU configuration (nanochat supports 1-8 GPUs)
NUM_GPUS_BASE = 4
NUM_GPUS_MID = 4
NUM_GPUS_SFT = 4
NUM_GPUS_RL = 4
NUM_GPUS_EVAL = 4
NUM_GPUS_TOKENIZER = 1
NUM_GPUS_INFERENCE = 1

WANDB_PROJECT_DEFAULT = "nanochat-modal"
BASE_DIR = "/data/.cache/nanochat"

# =============================================================================
# MODAL APP AND VOLUMES
# =============================================================================

app = App("nanochat-training")

data_volume = Volume.from_name("nanochat-data", create_if_missing=True)
checkpoint_volume = Volume.from_name("nanochat-checkpoints", create_if_missing=True)

VOLUME_CONFIG = {
    "/data": data_volume,
    "/checkpoints": checkpoint_volume,
}

# =============================================================================
# SECRETS SETUP
# =============================================================================

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

# =============================================================================
# CONTAINER IMAGE
# =============================================================================

NANOCHAT_IMAGE = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> $HOME/.bashrc",
    )
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> $HOME/.bashrc",
    )
    .add_local_dir(local_path="nanochat", remote_path="/root/nanochat", copy=True)
    .workdir("/root/nanochat")
    .run_commands(
        "bash -c 'source $HOME/.cargo/env && uv sync && uv run maturin develop --release --manifest-path rustbpe/Cargo.toml'"
    )
    .env(
        {
            "OMP_NUM_THREADS": "1",
            "NANOCHAT_BASE_DIR": "/data/.cache/nanochat",
            "HF_HOME": "/data/.cache/huggingface",
        }
    )
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def setup_base_dir():
    """Create base directory structure."""
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


# =============================================================================
# STAGE 1: DATASET DOWNLOAD
# =============================================================================


@app.function(
    image=NANOCHAT_IMAGE,
    volumes=VOLUME_CONFIG,
    timeout=2 * HOURS,
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
    max_chars: int = 2_000_000_000,
    vocab_size: int = 65536,
    doc_cap: int = 10000,
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


@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_TOKENIZER}",
    volumes=VOLUME_CONFIG,
    timeout=30 * MINUTES,
)
def evaluate_tokenizer():
    """Evaluate the trained tokenizer."""
    import subprocess

    print("=" * 80)
    print("EVALUATING TOKENIZER")
    print("=" * 80)

    result = subprocess.run(
        ["bash", "-c", "cd /root/nanochat && uv run python -m scripts.tok_eval"],
        capture_output=False,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Tokenizer evaluation failed with code {result.returncode}")

    print("\n" + "=" * 80)
    print("Tokenizer evaluation completed")
    print("=" * 80)

    return {"status": "completed"}


# =============================================================================
# STAGE 3: BASE MODEL PRETRAINING
# =============================================================================


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
    max_iterations: int = -1,
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

    eval_bundle_path = f"{BASE_DIR}/eval_bundle"
    if not os.path.exists(eval_bundle_path):
        print("Downloading eval bundle...")
        subprocess.run(
            [
                "curl",
                "-L",
                "-o",
                "eval_bundle.zip",
                "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip",
            ],
            check=True,
        )
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


@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_EVAL}",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def evaluate_base_model(max_per_task: int = 500):
    """Evaluate base model on CORE benchmark."""
    print("=" * 80)
    print("EVALUATING BASE MODEL - CORE METRIC")
    print("=" * 80)

    extra_args = []
    if max_per_task > 0:
        extra_args.append(f"--core_metric_max_per_task={max_per_task}")

    run_torchrun_command("scripts.base_eval", NUM_GPUS_EVAL, extra_args)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("Base model evaluation completed")
    print("=" * 80)

    return {"status": "completed"}


@app.function(
    image=NANOCHAT_IMAGE,
    gpu=f"{GPU_TYPE}:{NUM_GPUS_EVAL}",
    volumes=VOLUME_CONFIG,
    timeout=1 * HOURS,
)
def evaluate_base_loss():
    """Evaluate base model validation loss (bits per byte)."""
    print("=" * 80)
    print("EVALUATING BASE MODEL - VALIDATION LOSS")
    print("=" * 80)

    run_torchrun_command("scripts.base_loss", NUM_GPUS_EVAL)

    checkpoint_volume.commit()

    print("\n" + "=" * 80)
    print("Base loss evaluation completed")
    print("=" * 80)

    return {"status": "completed"}


# =============================================================================
# STAGE 4: MIDTRAINING
# =============================================================================


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


# =============================================================================
# STAGE 5: SUPERVISED FINE-TUNING (SFT)
# =============================================================================


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


# =============================================================================
# STAGE 6: REINFORCEMENT LEARNING (OPTIONAL)
# =============================================================================


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


# =============================================================================
# MAIN PIPELINE
# =============================================================================


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

    Pipeline stages:
    1. Download FineWeb dataset
    2. Train + evaluate tokenizer
    3. Train + evaluate base model
    4. Train + evaluate mid model
    5. Train + evaluate SFT model
    6. (Optional) Train + evaluate RL model
    7. Run final inference test

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
