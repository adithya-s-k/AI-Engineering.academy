import requests
import os
import argparse

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Upload PDF files to the server for conversion.")
    parser.add_argument('pdf_file_paths', type=str, nargs='+', help='The paths to the PDF files to be uploaded.')
    args = parser.parse_args()

    url = ""
    files = []
    
    for pdf_file_path in args.pdf_file_paths:
        with open(pdf_file_path, 'rb') as pdf_file:
            files.append(('pdf_files', (os.path.basename(pdf_file_path), pdf_file.read(), 'application/pdf')))
    
    params = {'extract_images': False}  # Optional parameter
    response = requests.post(url, files=files, params=params)

    response = response.json()
    
    markdown = response[0]["markdown"]
    
    with open("test.md", 'w', encoding="utf-8") as markdown_file:
        markdown_file.write(markdown)

if __name__ == "__main__":
    main()