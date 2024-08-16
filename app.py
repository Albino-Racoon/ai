import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import json
import os

# Set up FastAPI
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.DEBUG, filename='fastapi.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FileData(BaseModel):
    ime: str
    url: str

class FilesPayload(BaseModel):
    datoteke: List[FileData]

@app.post("/process_files")
async def process_files(payload: FilesPayload):
    temp_files_path = "temp_files"
    os.makedirs(temp_files_path, exist_ok=True)
    temp_data_file_path = os.path.join(temp_files_path, 'temp_data.json')

    try:
        # Save file information to a temporary JSON file
        with open(temp_data_file_path, 'w', encoding='utf-8') as f:
            json.dump([{"ime": d.ime, "url": os.path.join(temp_files_path, d.ime)} for d in payload.datoteke], f)
        
        logging.info(f"Saved temp data to {temp_data_file_path}")
        logging.info(f"Current working directory: {os.getcwd()}")

        # Ensure that environment variables are loaded correctly
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set.")
        
        logging.info("Environment variable OPENAI_API_KEY loaded successfully.")

        # Call the fine-tuning script
        result = subprocess.run(
            ["python3", "finetuning.py", temp_data_file_path],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            logging.error(f"Fine-tuning script failed with stdout: {result.stdout} and stderr: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Fine-tuning script failed with output: {result.stdout}")

        # Read model_info.json for the model_adapter_id
        model_info_file_path = 'model_info.json'
        try:
            with open(model_info_file_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                model_adapter_id = model_info.get('model_adapter_id')
                if not model_adapter_id:
                    raise ValueError("model_adapter_id not found in model_info.json")
                logging.info(f"Model adapter ID retrieved: {model_adapter_id}")
        except FileNotFoundError:
            logging.error(f"model_info.json file not found.")
            raise HTTPException(status_code=500, detail="model_info.json file not found.")
        except ValueError as ve:
            logging.error(f"Invalid data in model_info.json: {ve}")
            raise HTTPException(status_code=500, detail=str(ve))

        return {"status": "success", "modelAdapterId": model_adapter_id}

    except Exception as e:
        logging.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

