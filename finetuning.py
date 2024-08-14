import os
import sys
import json
import openai
import requests
import shutil
import docx
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

ACCESS_CORS = os.getenv('ACCESS_CORS')
FIREBASE_API_KEY = os.getenv('EXPRESS_APP_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

# Define file processing functions
def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return '\n'.join([para.text for para in doc.paragraphs])

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def download_file(url, destination):
    if os.path.isfile(url):
        if os.path.abspath(url) == os.path.abspath(destination):
            print(f"Source and destination are the same for {url}. Skipping copy.")
        else:
            shutil.copy(url, destination)
            print(f"Copied local file from {url} to {destination}.")
    elif url.startswith('http://') or url.startswith('https://'):
        response = requests.get(url)
        with open(destination, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file from {url} to {destination}.")
    else:
        raise ValueError(f"Invalid input: {url}. Must be a valid URL or a file path.")
    
    return destination

# Main function
def main(temp_file_path):
    print("Fine-tuning script started")

    print("Loading data...")
    with open(temp_file_path, 'r', encoding='utf-8') as file:
        datoteke = json.load(file)
    
    print(f"Received {len(datoteke)} files.")

    temp_dir = "temp_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Process files and extract content
    vsebina_datotek = []
    for datoteka in datoteke:
        file_path = download_file(datoteka['url'], os.path.join(temp_dir, datoteka['ime']))
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.txt':
            vsebina_datotek.append(extract_text_from_txt(file_path))
        elif ext == '.docx':
            vsebina_datotek.append(extract_text_from_docx(file_path))
        elif ext == '.pdf':
            vsebina_datotek.append(extract_text_from_pdf(file_path))
        else:
            print(f"Skipping unsupported file type: {ext}")
            continue

    # Check if there is any content to fine-tune
    if not vsebina_datotek:
        print("No valid files to process.")
        return

    # Prepare content for fine-tuning
    celotna_vsebina = "\n".join(vsebina_datotek)
    deli_vsebine = [celotna_vsebina[i:i+1000] for i in range(0, len(celotna_vsebina), 1000)]

    samples = []
    for del_vsebine in deli_vsebine:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Updated to a supported model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": del_vsebine}
            ],
            max_tokens=512,
            temperature=0.7
        )
        generated_output = response['choices'][0]['message']['content']
        questions_answers = generated_output.split("\n\n")
        for qa in questions_answers:
            if ':' in qa:
                question, answer = qa.split(':', 1)
                if question.strip() and answer.strip():
                    samples.append({
                        "prompt": f"### Instruction: {question.strip()}\n\n###\n\n",
                        "completion": f" {answer.strip()}\n"
                    })

    print(f"Generated {len(samples)} samples.")

    # Save samples to JSONL file
    if samples:
        with open('fine_tune_data.jsonl', 'w') as f:
            for item in samples:
                json.dump(item, f)
                f.write('\n')
        print(f"Data for fine-tuning saved to 'fine_tune_data.jsonl'")

        # Upload the file for fine-tuning and get the file ID
        print("Uploading file for fine-tuning...")
        file_response = openai.File.create(
            file=open("fine_tune_data.jsonl", "rb"),
            purpose='fine-tune'
        )

        file_id = file_response['id']
        print(f"Uploaded file ID: {file_id}")

        # Start the fine-tuning process
        print("Starting fine-tuning process...")
        try:
            fine_tune_response = openai.FineTune.create(
                training_file=file_id,
                model="gpt-3.5-turbo-1106",  # Use the latest model that supports fine-tuning
                suffix="my-experiment",
                n_epochs=4,  # Adjust based on your dataset size
                learning_rate_multiplier=0.1,  # Auto-tune this or use a default
                batch_size="auto"  # Auto-batch size for optimal training
            )
            print(f"Fine-tune job started: {fine_tune_response['id']}")

            # Monitor the fine-tuning process
            fine_tune_id = fine_tune_response['id']
            status = None
            while status not in ["succeeded", "failed"]:
                fine_tune_status = openai.FineTune.retrieve(id=fine_tune_id)
                status = fine_tune_status['status']
                print(f"Status: {status}")
                if status in ["succeeded", "failed"]:
                    break

            if status == "succeeded":
                print(f"Fine-tuning succeeded. Model ID: {fine_tune_status['fine_tuned_model']}")
            else:
                print("Fine-tuning failed.")

        except openai.error.InvalidRequestError as e:
            print(f"Error during fine-tuning: {e}")
            print("Please check the OpenAI API documentation for the correct endpoint and model support.")
            sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print("You need to provide the path to the temporary data file.")
