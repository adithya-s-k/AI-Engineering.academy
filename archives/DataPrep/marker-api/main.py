import asyncio
import aiohttp
import random

async def invoke_convert_endpoint(file_path):
    async with aiohttp.ClientSession() as session:
        # await asyncio.sleep(random.uniform(0, 5))  # Random delay between 0 to 5 seconds
        async with session.post("http://172.210.48.84:8000/convert", data={"pdf_files": open(file_path, 'rb')}) as response:
            try:
                result = await response.json()
                print(result[0]["time"])
            except:
                print("Failed to convert")

async def main():
    file_paths = ["test1.pdf" for i in range(5)]
    print(len(file_paths))
    tasks = [invoke_convert_endpoint(file_path) for file_path in file_paths]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
