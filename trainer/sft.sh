#!/bin/bash

# Set the necessary environment variables if needed
export CUDA_VISIBLE_DEVICES=0  # Set the GPU device if using GPU

# Define the paths and arguments
PYTHON_SCRIPT="your_python_script.py"
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DATASET_NAME="lvwerra/stack-exchange-paired"
SUBSET="data/finetune"
SPLIT="train"
SIZE_VALID_SET=4000
STREAMING=true
SHUFFLE_BUFFER=5000
SEQ_LENGTH=1024
NUM_WORKERS=4
PACKING=true
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_R=8
GRADIENT_CHECKPOINTING=false
OUTPUT_DIR="./output"

# Run the Python script with arguments
python $PYTHON_SCRIPT \
    --model_name $MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --subset $SUBSET \
    --split $SPLIT \
    --size_valid_set $SIZE_VALID_SET \
    --streaming $STREAMING \
    --shuffle_buffer $SHUFFLE_BUFFER \
    --seq_length $SEQ_LENGTH \
    --num_workers $NUM_WORKERS \
    --packing $PACKING \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --lora_r $LORA_R \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --output_dir $OUTPUT_DIR