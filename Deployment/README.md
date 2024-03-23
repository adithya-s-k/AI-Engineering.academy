### FastAPI Streaming Server

This repository contains code for setting up a FastAPI streaming server using the Transformers library. The server can generate streaming responses based on user queries using a pre-trained language model.

### Getting Started

To get started with the streaming server, follow the steps below.

#### Prerequisites

Make sure you have Python installed on your system. You can download it from [here](https://www.python.org/downloads/).

#### Installation


Install the required dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Running the Server

To start the streaming server, run the following command:

```bash
python server.py [--model_id MODEL_ID] [--quantization] [--port PORT] [--host HOST] [--max_new_tokens MAX_NEW_TOKENS]
```

#### Optional Arguments

- `--model_id MODEL_ID`: ID of the model to use. Default is `"mistralai/Mistral-7B-Instruct-v0.2"`.
- `--quantization`: Whether to use quantization or not. Default is `False`.
- `--port PORT`: Port number to run the server on. Default is `8000`.
- `--host HOST`: Host address to bind the server to. Default is `"127.0.0.1"`.
- `--max_new_tokens MAX_NEW_TOKENS`: Maximum number of new tokens for generation. Default is `1024`.



### Client Usage

To query the streaming server and receive streaming responses, you can use the provided client script.

#### Client Script

The client script allows you to send queries to the streaming server and receive streaming responses. You can run it using the following command:

```bash
python client.py --endpoint <server_endpoint> --query <query_string>
```

#### Optional Arguments

- `--endpoint`: URL of the FastAPI endpoint. Default is `http://127.0.0.1:8000/query-stream`.
- `--query`: Query to send to the endpoint. Default is `"give me the recipe for chicken butter masala in detail"`.

### Example

```bash
python client.py --endpoint http://127.0.0.1:8000/query-stream --query "What is the weather today?"
```

### Additional Notes

- The server can be customized with additional optional arguments. These arguments can be found in the `server.py` file and can be passed via command line arguments when running the server script.
- Make sure the server is running before executing the client script to receive streaming responses.



## Deploying LLMs on GPUs

This directory contains the scripts and instructions for deploying LLMs on GPUs.

For the following Tutorial we will be deploying [Mistral Instruct Model 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

### Deploying using [VLLM](https://docs.vllm.ai/en/latest/)

There are two ways to deploy LLMs using VLLM either through a docker image or via a python package

#### Deploying using Docker:

Pre-requisites:
    - Docker should be installed
    - A GPU with atleast 16GB of vRAM(T4 , V100 , A100)



#### Deploying using Python Package:

Pre-requisites:
    - Python 3.9 or higher
    - A GPU with atleast 16GB of vRAM(T4 , V100 , A100)

### Deploying using Text Generation Inferece(TGi)

#### Deploying using Text Generation Inferece using Docker


#### Testing out Inference from the models

- Check out the [notebook](https://github.com/adithya-s-k/LLM-Cookbook/blob/main/Deployment/inference.ipynb)


