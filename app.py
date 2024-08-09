import os
import sys
import json
import requests
import PyPDF2
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
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded file from {url} to {destination}.")
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
        print(f"Napaka pri branju PDF datoteke: {e}")
        return ""

@app.post("/process_files")
async def process_files(payload: FilesPayload):
    temp_files_path = "temp_files"
    os.makedirs(temp_files_path, exist_ok=True)

    datoteke = payload.datoteke
    temp_data_file_path = os.path.join(temp_files_path, 'temp_data.json')
    
    # Download files from URLs and save to disk
    downloaded_files = []
    for datoteka in datoteke:
        file_path = os.path.join(temp_files_path, datoteka.ime)
        download_file(datoteka.url, file_path)
        downloaded_files.append(file_path)

    # Save file information to a temporary JSON file
    with open(temp_data_file_path, 'w', encoding='utf-8') as f:
        json.dump([{"ime": d.ime, "url": file_path} for d, file_path in zip(datoteke, downloaded_files)], f)

    # Fine-tuning and output generation logic can go here
    print(f"Processed {len(datoteke)} files. Results stored in {temp_data_file_path}.")
    
    return {"status": "success", "message": "Datoteke so bile obdelane."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
