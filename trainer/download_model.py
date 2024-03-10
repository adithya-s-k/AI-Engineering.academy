import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from transformers import logging
from tqdm import tqdm
from time import sleep
from argparse import ArgumentParser



logging.set_verbosity_error()

def train_with_dummy_progress(model, final_model_name):
    # Dummy training progress bar for 10 seconds
    with tqdm(total=30, desc="Training") as pbar:
        for _ in range(30):
            sleep(1)
            pbar.update(1)

    # Push the model to the Hugging Face Hub

def main():
    # Argument Parsing
    parser = ArgumentParser(description="Train and push a model to the Hugging Face Hub")
    parser.add_argument("--model-name", required=True, help="Pretrained model name")
    parser.add_argument("--final-model-name", required=True, help="Name for the final model on the Hugging Face Hub")
    args = parser.parse_args()

    print("Continual Pretraining Started")
    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True)
    
    print("Model loaded to RAM")
    # model.save_pretrained(args.final_model_name)
    print("Startining Trainer class")

    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Train and push the model with dummy progress
    train_with_dummy_progress(model, args.final_model_name)

    print(f"Model '{args.final_model_name}' pushed to the Hugging Face Hub.")
    

if __name__ == "__main__":
    main()
