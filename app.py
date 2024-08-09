import os
import json
import requests
import shutil
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import PyPDF2
import random
from gradientai import Gradient  # Assuming you have this library for fine-tuning

app = FastAPI()

class FileData(BaseModel):
    ime: str
    url: str

class FilesPayload(BaseModel):
    datoteke: List[FileData]

def download_file(url, destination):
    if os.path.exists(destination):
        print(f"File {destination} already exists, skipping download.")
        return destination
    else:
        if os.path.isfile(url):
            shutil.copy(url, destination)
            print(f"Copied local file from {url} to {destination}.")
        else:
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

def generate_evaluation_prompt(actual_response, predicted_response):
    return f"Given the expected response: '{actual_response}', and the generated response: '{predicted_response}', does the generated response accurately capture the key information? Yes or No."

def evaluate_response_with_model(model_adapter, prompt):
    evaluation_result = model_adapter.complete(query=prompt, max_generated_token_count=100).generated_output
    return "Yes" in evaluation_result

def parse_instruction_response(inputs):
    parts = inputs.split("#### Response:")
    instruction = parts[0].strip().replace("### Instruction:", "").strip()
    response = parts[1].strip() if len(parts) > 1 else ""
    return instruction, response

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
    
    downloaded_files = [file for file in downloaded_files if file is not None]
    vsebina_datotek = [preberi_pdf(file_path) for file_path in downloaded_files]
    celotna_vsebina = "\n".join(vsebina_datotek)
    deli_vsebine = [celotna_vsebina[i:i+1000] for i in range(0, len(celotna_vsebina), 1000)]

    url = "https://api.gradient.ai/api/models/399e5ea8-21ba-4558-89b3-d962f7efd0db_model_adapter/complete"
    headers = {
        "accept": "application/json",
        "x-gradient-workspace-id": "86abdbb7-ca5f-4f71-9882-01970e111de7_workspace",
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ['GRADIENT_ACCESS_TOKEN']}"
    }

    samples = []

    print("Sending requests to Gradient API...")
    for del_vsebine in deli_vsebine:
        payload = {
            "autoTemplate": True,
            "query": del_vsebine,
            "maxGeneratedTokenCount": 511
        }
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            generated_output = data.get("generatedOutput")
            questions_answers = generated_output.split("\n\n")
            for qa in questions_answers:
                if ':' in qa:
                    question, answer = qa.split(':', 1)
                    if question.strip() and answer.strip():
                        sample = {
                            "inputs": f"""
                                ### Instruction: {question.strip()}
                                #### Response: {answer.strip()}
                            """
                        }
                        samples.append(sample)
        else:
            print(f"Error in request: {response.status_code}, {response.text}")

    print(f"Generated {len(samples)} samples.")

    if samples:
        with Gradient() as gradient:
            base_model = gradient.get_base_model(base_model_slug="llama2-7b-chat")
            new_model_adapter = base_model.create_model_adapter(name="nadgrajen")
            print(f"Created model adapter with ID: {new_model_adapter.id}")

            random_sample = random.choice(samples)
            instruction, actual_response = parse_instruction_response(random_sample["inputs"])
            sample_query = f"{instruction}\n\n#### Response:"
            print(f"Question: {sample_query}")

            completion_before = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            print(f"Generated (before fine-tuning): {completion_before}")

            num_epochs = 3
            success = False
            count = 0
            while count < num_epochs and not success:
                print(f"Fine-tuning model, iteration {count + 1}")
                new_model_adapter.fine_tune(samples=samples)

                completion_after = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
                print(f"Generated (after fine-tuning iteration {count + 1}): {completion_after}")

                evaluation_prompt = generate_evaluation_prompt(actual_response, completion_after)
                success = evaluate_response_with_model(base_model, evaluation_prompt)
                print("Evaluation success:", "Yes" if success else "No")
                count += 1

            print(f"Model Adapter ID: {new_model_adapter.id}")
    else:
        print("No samples generated for fine-tuning.")

@app.post("/process_files")
async def process_files(payload: FilesPayload):
    temp_files_path = "temp_files"
    os.makedirs(temp_files_path, exist_ok=True)

    datoteke = payload.datoteke
    temp_data_file_path = os.path.join(temp_files_path, 'temp_data.json')

    for datoteka in datoteke:
        file_path = os.path.join(temp_files_path, datoteka.ime)
        download_file(datoteka.url, file_path)

    with open(temp_data_file_path, 'w', encoding='utf-8') as f:
        json.dump([{"ime": d.ime, "url": os.path.join(temp_files_path, d.ime)} for d in datoteke], f)

    run_finetuning(temp_data_file_path)

    return {"status": "success", "message": "Files processed successfully."}
