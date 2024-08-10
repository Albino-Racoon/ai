import os
import sys
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import shutil

app = FastAPI()


class FileData(BaseModel):
    ime: str
    url: str

class FilesPayload(BaseModel):
    datoteke: List[FileData]

def download_file(url, destination):
    try:
        if os.path.isfile(url):
            shutil.copy(url, destination)
            print(f"Copied local file from {url} to {destination}.")
        elif url.startswith('http://') or url.startswith('https://'):
            response = requests.get(url)
            with open(destination, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded file from {url} to {destination}.")
        else:
            raise ValueError(f"Invalid input: {url}. Must be a valid URL or a file path.")
    except Exception as e:
        print(f"Error downloading file from {url} to {destination}: {e}")
        raise e

@app.post("/process_files")
async def process_files(payload: FilesPayload):
    try:
        temp_files_path = "temp_files"
        os.makedirs(temp_files_path, exist_ok=True)

        datoteke = payload.datoteke
        temp_data_file_path = os.path.join(temp_files_path, 'temp_data.json')

        # Download files from URLs and save to disk
        for datoteka in datoteke:
            file_path = os.path.join(temp_files_path, datoteka.ime)
            try:
                download_file(datoteka.url, file_path)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading file {datoteka.ime}: {e}")

        # Save file information to a temporary JSON file
        try:
            with open(temp_data_file_path, 'w', encoding='utf-8') as f:
                json.dump([{"ime": d.ime, "url": os.path.join(temp_files_path, d.ime)} for d in datoteke], f)
            print('temp_data.json has been updated.')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating temp_data.json: {e}")

        # Call the fine-tuning script
        try:
            result = os.system(f"python3 finetuning.py {temp_data_file_path}")
            model_info_file_path = os.path.join(temp_files_path, 'model_info.json')
    
            if result != 0:
                raise HTTPException(status_code=500, detail="Fine-tuning script failed")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error running fine-tuning script: {e}")

        return {"status": "success", "message": "Files processed successfully."}
    
    except HTTPException as e:
        print(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        process_files(sys.argv[1])
    else:
        print("Potrebno je podati pot do začasne datoteke.")
