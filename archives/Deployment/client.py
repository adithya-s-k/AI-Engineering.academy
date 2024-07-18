import argparse
import requests
import time

# Initialize argument parser
parser = argparse.ArgumentParser(description="Query a FastAPI endpoint for streaming response")
parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000/query-stream", help="URL of the FastAPI endpoint")
parser.add_argument("--query", type=str, default="give me the recipe for chicken butter masala in detail", help="Query to send to the endpoint")

# Parse arguments
args = parser.parse_args()

endpoint = args.endpoint
query = args.query

# Invoke the FastAPI endpoint
response = requests.get(f"{endpoint}/?query=" + query, stream=True)

# Print the streaming response on the same line
for chunk in response.iter_content(chunk_size=None):
    if chunk:
        print(chunk.decode('utf-8'), end='', flush=True)  # Use end='' to print on the same line
    time.sleep(0.05)  # Adjust sleep time as needed
