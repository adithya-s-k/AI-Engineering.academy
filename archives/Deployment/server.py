import asyncio
import logging  # Import the logging module
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import torch
from queue import Queue
from threading import Thread
import argparse

# Initialize argument parser
parser = argparse.ArgumentParser(description="FastAPI application with argument parsing")
parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help="ID of the model to use")
parser.add_argument("--quantization", action="store_true", help="Whether to use quantization or not")
parser.add_argument("--port", type=int, default=8000, help="Port number to run the server on")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server to")
parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens for generation")

# Parse arguments
args = parser.parse_args()

model_id = args.model_id
quantization = args.quantization
port = args.port
host = args.host
max_new_tokens = args.max_new_tokens

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting server...")
device = "cuda:0"


if quantization:
    logger.info("Loading quantized model...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, 
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=device,
        )
else:
    logger.info("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device,
        )



class CustomStreamer(TextStreamer):
    def __init__(self, queue, tokenizer, skip_prompt, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self._queue = queue
        self.stop_signal = None
        self.timeout = 1
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        self._queue.put(text)
        if stream_end:
            self._queue.put(self.stop_signal)

app = FastAPI()

streamer_queue = Queue()
streamer = CustomStreamer(streamer_queue, tokenizer, True)

@app.get('/query-stream/')
async def stream(query: str):
    logger.info(f'Query received: {query}')  # Log the received query
    
    inputs = tokenizer([query], return_tensors="pt").to("cuda:0")
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_new_tokens, temperature=0.1)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    async def response_generator():
        while True:
            value = streamer_queue.get()
            if value is None:
                break
            yield value
            streamer_queue.task_done()
            await asyncio.sleep(0.1)

    return StreamingResponse(response_generator(), media_type='text/event-stream')

if __name__ == "__main__":
    logger.info(f"Server listening on http://{host}:{port}")
    import uvicorn
    uvicorn.run("server:app", host=host, port=port)
