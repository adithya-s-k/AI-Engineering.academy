import os
import argparse
import glob
from azure.storage.blob import BlobServiceClient

def download_files_from_blob_storage(connection_string, folder_name, local_folder_path, container_name):
    try:
        print("Creating BlobServiceClient...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        print("BlobServiceClient created successfully.")

        print("Creating ContainerClient...")
        container_client = blob_service_client.get_container_client(container_name)
        print("ContainerClient created successfully.")

        print("Listing blobs in the container...")
        blob_list = container_client.list_blobs(name_starts_with=folder_name)

        print("Downloading files from Azure Blob Storage...")
        for blob in blob_list:
            blob_name = blob.name
            blob_client = container_client.get_blob_client(blob_name)
            download_file_path = os.path.join(local_folder_path, blob_name.replace("/", "\\"))
            os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
            with open(download_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            print(f"Downloaded {blob_name} to {download_file_path}")

        print("Files downloaded from Azure Blob Storage.")

        # print("Combining all text files into a single file named corpus.txt...")
        # corpus_file_path = os.path.join(local_folder_path, "corpus.txt")
        # with open(corpus_file_path, "w") as corpus_file:
        #     for text_file in glob.glob(os.path.join(local_folder_path, "*.txt")):
        #         with open(text_file, "r") as file:
        #             corpus_file.write(file.read())
        # print("All text files combined into corpus.txt.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from Azure Blob Storage and combine text files.')
    parser.add_argument('--connection-string', type=str, required=True, help='Azure Blob Storage connection string')
    parser.add_argument('--folder-name', type=str, required=True, help='Folder name in Azure Blob Storage to download files from')
    parser.add_argument('--local-folder-path', default="corpus", type=str, help='Local folder path to download files to')
    parser.add_argument('--container-name', default="data-corpus", type=str, help='Azure Blob Storage container name')

    args = parser.parse_args()

    download_files_from_blob_storage(args.connection_string, args.folder_name, args.local_folder_path, args.container_name)
