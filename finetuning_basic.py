import openai
import os
import time

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to upload the training file
def upload_training_file(file_path):
    with open(file_path, "rb") as f:
        response = openai.File.create(file=f, purpose='fine-tune')
    return response['id']

# Function to create a fine-tuning job
def create_fine_tune(training_file_id, model="gpt-3.5-turbo"):
    response = openai.FineTune.create(
        training_file=training_file_id,
        model=model
    )
    return response['id']

# Function to monitor the fine-tuning job
def monitor_fine_tune(fine_tune_id):
    while True:
        status = openai.FineTune.retrieve(fine_tune_id)
        if status['status'] == 'succeeded':
            print(f"Fine-tuning succeeded. Model ID: {status['fine_tuned_model']}")
            break
        elif status['status'] == 'failed':
            print("Fine-tuning failed.")
            break
        else:
            print(f"Fine-tuning in progress... (Status: {status['status']})")
            time.sleep(60)  # Wait for 60 seconds before checking again

# Function to analyze the fine-tuned model
def analyze_fine_tuned_model(fine_tuned_model_id, prompt):
    response = openai.Completion.create(
        model=fine_tuned_model_id,
        prompt=prompt,
        max_tokens=50
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    # Path to your dataset in JSONL format
    data_file_path = r"C:\Users\jasar\Desktop\secret_deploy\ai\sample_data.jsonl"

    # Step 1: Upload your training file
    training_file_id = upload_training_file(data_file_path)
    print(f"Uploaded training file. ID: {training_file_id}")

    # Step 2: Create a fine-tuning job
    fine_tune_id = create_fine_tune(training_file_id)
    print(f"Fine-tuning job created. ID: {fine_tune_id}")

    # Step 3: Monitor the fine-tuning process
    monitor_fine_tune(fine_tune_id)

    # Step 4: Analyze the fine-tuned model
    fine_tuned_model_id = fine_tune_id  # Replace with actual fine-tuned model ID from step 3
    prompt = "What is the capital of Japan?"
    result = analyze_fine_tuned_model(fine_tuned_model_id, prompt)
    print(f"Model's response: {result}")
