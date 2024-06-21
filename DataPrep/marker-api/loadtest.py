import asyncio
import aiohttp
import random
import time
from datetime import datetime
import json
import os

# Function to invoke the convert endpoint
async def invoke_convert_endpoint(session, file_path):
    try:
        async with session.post("http://172.210.48.84:8000/convert", data={"pdf_files": open(file_path, 'rb')}) as response:
            result = await response.json()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"File: {file_path}, Status: {response.status}, Result: {result}, Timestamp: {timestamp}")
            return {"file": file_path, "status": response.status, "result": result, "timestamp": timestamp}
    except Exception as e:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Failed to send {file_path}: {e}, Timestamp: {timestamp}")
        return {"file": file_path, "status": "failed", "error": str(e), "timestamp": timestamp}

# Function to run the load test
async def load_test(file_paths, duration_minutes=1):
    end_time = time.time() + duration_minutes * 60
    results = []

    async with aiohttp.ClientSession() as session:
        while time.time() < end_time:
            file_path = random.choice(file_paths)
            task = invoke_convert_endpoint(session, file_path)
            result = await task
            results.append(result)

            # Save each result to a JSON file
            file_name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(file_name, 'w') as f:
                json.dump(result, f, indent=4)

            # Random interval between 0.1 and 2 seconds
            await asyncio.sleep(random.uniform(0.1, 2))

    return results

# Main function to initiate the load test
async def main():
    file_paths = [f"{i}.pdf" for i in range(1, 11)]
    results = await load_test(file_paths, duration_minutes=1)
    
    # Generate report
    report = {}
    for result in results:
        file = result["file"]
        if file not in report:
            report[file] = {"success": 0, "failed": 0, "details": []}
        if result["status"] == 200:
            report[file]["success"] += 1
        else:
            report[file]["failed"] += 1
        report[file]["details"].append(result)
    
    print("\nLoad Test Report:")
    for file, data in report.items():
        print(f"\nFile: {file}")
        print(f"Success: {data['success']}, Failed: {data['failed']}")
        for detail in data["details"]:
            print(f"  {detail['timestamp']}: Status {detail['status']}, Result: {detail.get('result', detail.get('error', 'No result'))}")

if __name__ == "__main__":
    asyncio.run(main())
