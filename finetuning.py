import os
import sys
import json
import requests
import PyPDF2
import random
import shutil
import datetime
from gradientai import Gradient

def download_file(url, destination):
    # Check if the input is a URL or a local file path
    if os.path.isfile(url):
        if os.path.abspath(url) == os.path.abspath(destination):
            print(f"Source and destination are the same for {url}. Skipping copy.")
        else:
            # If the input is a local file path, copy it directly
            shutil.copy(url, destination)
            print(f"Copied local file from {url} to {destination}.")
    elif url.startswith('http://') or url.startswith('https://'):
        # If it's a URL, download the file
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file from {url} to {destination}.")
    else:
        raise ValueError(f"Invalid input: {url}. Must be a valid URL or a file path.")
    
    return destination


def main(temp_file_path):
    print("Finetuning script started")
    os.environ['GRADIENT_ACCESS_TOKEN'] = "zHkm0nTvAVXsUobrgw4UelOfRQsKRCl2"
    os.environ['GRADIENT_WORKSPACE_ID'] = "86abdbb7-ca5f-4f71-9882-01970e111de7_workspace"

    print("Loading data...")
    with open(temp_file_path, 'r', encoding='utf-8') as file:
        datoteke = json.load(file)
    
    print(f"Received {len(datoteke)} files.")

    def preberi_pdf(file_path):
        try:
            print(f"Reading file: {file_path}")
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                vsebina = "".join([page.extract_text() for page in reader.pages])
                return vsebina
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""

    def parse_instruction_response(inputs):
        parts = inputs.split("#### Response:")
        instruction = parts[0].strip().replace("### Instruction:", "").strip()
        response = parts[1].strip() if len(parts) > 1 else ""
        return instruction, response

    def generate_evaluation_prompt(actual_response, predicted_response):
        return f"Given the expected response: '{actual_response}', and the generated response: '{predicted_response}', does the generated response accurately capture the key information? Yes or No."

    def evaluate_response_with_model(model_adapter, prompt):
        evaluation_result = model_adapter.complete(query=prompt, max_generated_token_count=100).generated_output
        return "Yes" in evaluation_result

    print("Reading file content...")
    temp_dir = "temp_files"
    downloaded_files = [download_file(datoteka['url'], os.path.join(temp_dir, datoteka['ime'])) for datoteka in datoteke]
    vsebina_datotek = [preberi_pdf(file_path) for file_path in downloaded_files]

    celotna_vsebina = "\n".join(vsebina_datotek)
    deli_vsebine = [celotna_vsebina[i:i+1000] for i in range(0, len(celotna_vsebina), 1000)]

    url = "https://api.gradient.ai/api/models/399e5ea8-21ba-4558-89b3-d962f7efd0db_model_adapter/complete"
    headers = {
        "accept": "application/json",
        "x-gradient-workspace-id": os.environ['GRADIENT_WORKSPACE_ID'],
        "content-type": "application/json",
        "authorization": f"Bearer {os.environ['GRADIENT_ACCESS_TOKEN']}"
    }

    samples = []

    print("Sending requests to Gradient API...")
    for del_vsebine in deli_vsebine:
        payload = {"autoTemplate": True, "query": del_vsebine, "maxGeneratedTokenCount": 511}
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            generated_output = data.get("generatedOutput", "")
            questions_answers = generated_output.split("\n\n")
            for qa in questions_answers:
                if ':' in qa:
                    question, answer = qa.split(':', 1)
                    if question.strip() and answer.strip():
                        sample = {"inputs": f"### Instruction: {question.strip()}\n#### Response: {answer.strip()}"}
                        samples.append(sample)
        else:
            print(f"Request error: {response.status_code}, {response.text}")

    print(f"Generated {len(samples)} samples.")

    if samples:
        with Gradient() as gradient:
            base_model = gradient.get_base_model(base_model_slug="llama2-7b-chat")
            new_model_adapter = base_model.create_model_adapter(name="fine_tuned_model")
            print(f"Model adapter created with ID: {new_model_adapter.id}")

            random_sample = random.choice(samples)
            instruction, actual_response = parse_instruction_response(random_sample["inputs"])
            sample_query = f"{instruction}\n\n#### Response:"
            print(f"Sample Query: {sample_query}")

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
                print(f"Evaluation successful: {'Yes' if success else 'No'}")
                count += 1

            # Save the model ID to a JSON file
            model_info = {
                "model_adapter_id": new_model_adapter.id,
                "creation_time": str(datetime.datetime.now())

            }
            with open("model_info.json", "w") as f:
                json.dump(model_info, f)
            print(f"Model Adapter ID: {new_model_adapter.id} saved to model_info.json")
    else:
        print("No samples were generated for fine-tuning.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("You need to provide the path to the temporary data file.")
