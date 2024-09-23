import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, TextStreamer
import argparse

def generate_output(model, tokenizer, instruction):
    batch = tokenizer(instruction, return_tensors="pt", add_special_tokens=True)

    print("=" * 40)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.1,
            max_new_tokens=1024,
            temperature=0.9,
            top_p=0.95,
            top_k=40,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            padding_side='left'
        )
        streamer = TextStreamer(tokenizer)
        generated = model.generate(
            inputs=batch["input_ids"].to("cuda"),
            generation_config=generation_config,
            streamer=streamer,
        )
    print("=" * 40)
    print(tokenizer.decode(generated["sequences"].cpu().tolist()[0]))

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate text with LlamaForCausalLM")
    parser.add_argument("--prompt", type=str, required=True, help="The instruction prompt")
    parser.add_argument("--model", type=str, required=True, help="The model ID")

    args = parser.parse_args()

    # Load the specified model and tokenizer
    model = LlamaForCausalLM.from_pretrained(args.model, device_map={"": 0})
    tokenizer = LlamaTokenizer.from_pretrained(args.model, add_eos_token=True, padding_side="left")

    # Generate output using the provided prompt
    generate_output(model, tokenizer, args.prompt)
