import os
import sys
import json
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class FileData(BaseModel):
    ime: str
    url: str

class FilesPayload(BaseModel):
    datoteke: List[FileData]

@app.post("/process_files")
async def process_files(payload: FilesPayload):
    temp_files_path = "temp_files"
    os.makedirs(temp_files_path, exist_ok=True)

    datoteke = payload.datoteke
    temp_data_file_path = os.path.join(temp_files_path, 'temp_data.json')
    
    # Download files from URLs and save to disk
    for datoteka in datoteke:
        file_path = os.path.join(temp_files_path, datoteka.ime)
        try:
            with open(file_path, 'wb') as f:
                f.write(requests.get(datoteka.url).content)
            print(f"File {datoteka.ime} saved at {file_path}.")
        except Exception as e:
            print(f"Error saving file {datoteka.ime}: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving file {datoteka.ime}")

    # Save file information to a temporary JSON file
    try:
        with open(temp_data_file_path, 'w', encoding='utf-8') as f:
            json.dump([{"ime": d.ime, "url": os.path.join(temp_files_path, d.ime)} for d in datoteke], f)
        print('temp_data.json has been updated.')
    except Exception as e:
        print(f"Error updating temp_data.json: {e}")
        raise HTTPException(status_code=500, detail="Error updating temp_data.json")

    # Call the fine-tuning script
    try:
        result = os.system(f"python finetuning.py {temp_data_file_path}")
        if result != 0:
            raise HTTPException(status_code=500, detail="Fine-tuning script failed")
    except Exception as e:
        print(f"Error running fine-tuning script: {e}")
        raise HTTPException(status_code=500, detail="Error running fine-tuning script")

    return {"status": "success", "message": "Files processed successfully."}
