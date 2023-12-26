
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


