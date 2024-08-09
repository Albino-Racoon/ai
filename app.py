import os
import sys
import json
import requests
import shutil
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class FileData(BaseModel):
    ime: str
    url: str

class FilesPayload(BaseModel):
    datoteke: List[FileData]

def download_file(url, destination):
    if os.path.isfile(url):
        # If the input is a local file path, copy it directly
        shutil.copy(url, destination)
        print(f"Copied local file from {url} to {destination}.")
    else:
        # If it's a URL, download the file
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(destination, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded file from {url} to {destination}.")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")
            return None
    return destination

def preberi_pdf(file_path):
    try:
        print(f"Reading file: {file_path}")
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            vsebina = ""
            for stran in reader.pages:
                vsebina += stran.extract_text()
            return vsebina
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return ""

def run_finetuning(temp_file_path):
    print("Finetuning script started")
    os.environ['GRADIENT_ACCESS_TOKEN'] = "zHkm0nTvAVXsUobrgw4UelOfRQsKRCl2"
    os.environ['GRADIENT_WORKSPACE_ID'] = "86abdbb7-ca5f-4f71-9882-01970e111de7_workspace"

    print("Loading data...")
    with open(temp_file_path, 'r', encoding='utf-8') as file:
        datoteke = json.load(file)
    
    print(f"Received {len(datoteke)} files.")

    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    downloaded_files = [
        download_file(datoteka['url'], os.path.join(temp_dir, datoteka['ime'])) 
        for datoteka in datoteke
    ]
    
    # Filter out any None values from failed downloads
    downloaded_files = [file for file in downloaded_files if file is not None]

    vsebina_datotek = [preberi_pdf(file_path) for file_path in downloaded_files]

    # Here, add the fine-tuning logic...

    print(f"Processed {len(vsebina_datotek)} files. Results stored in {temp_file_path}.")

@app.post("/process_files")
async def process_files(payload: FilesPayload):
    temp_files_path = "temp_files"
    os.makedirs(temp_files_path, exist_ok=True)

    datoteke = payload.datoteke
    temp_data_file_path = os.path.join(temp_files_path, 'temp_data.json')

    # Download files from URLs and save to disk
    for datoteka in datoteke:
        file_path = os.path.join(temp_files_path, datoteka.ime)
        with open(file_path, 'wb') as f:
            f.write(requests.get(datoteka.url).content)
        print(f"File {datoteka.ime} saved at {file_path}.")

    # Save file information to a temporary JSON file
    with open(temp_data_file_path, 'w', encoding='utf-8') as f:
        json.dump([{"ime": d.ime, "url": os.path.join(temp_files_path, d.ime)} for d in datoteke], f)

    # Run the finetuning process
    run_finetuning(temp_data_file_path)

    return {"status": "success", "message": "Files processed successfully."}
