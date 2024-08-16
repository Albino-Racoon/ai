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
        logging.info(f"Running finetuning.py with temp_data_file_path: {temp_data_file_path}")
        result = subprocess.run(
            ["python3", "finetuning.py", temp_data_file_path],
            capture_output=True, text=True
        )

        logging.info(f"Subprocess completed with return code: {result.returncode}")
        logging.debug(f"Subprocess stdout: {result.stdout}")
        logging.debug(f"Subprocess stderr: {result.stderr}")

        if result.returncode != 0:
            logging.error(f"Fine-tuning script failed with stdout: {result.stdout} and stderr: {result.stderr}")
            raise HTTPException(status_code=500, detail=f"Fine-tuning script failed with output: {result.stdout}")

        # Parse the model ID from the script output (assuming it's printed as last line)
        last_line = result.stdout.strip().split("\n")[-1]
        model_id = last_line if last_line.startswith("model-") else None

        if not model_id:
            raise ValueError("Model ID not found in the script output.")

        logging.info(f"Fine-tuned model ID: {model_id}")

        return {"status": "success", "modelAdapterId": model_id}

    except Exception as e:
        logging.error(f"Error processing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
