import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Specify the file to upload and its purpose
file_path = "local_storage/finetuned/combined_fine_tune_data.jsonl"  # Path to your file
purpose = "fine-tune"  # Specify the purpose of the file (e.g., "fine-tune")

try:
    # Upload the file
    response = openai.files.create(
        file=open(file_path, "rb"),
        purpose=purpose
    )
    # Print the response
    print(f"File uploaded successfully: {response}")
except Exception as e:
    print(f"Error uploading file: {e}")

try:
    # Create fine-tuning job
    job = openai.fine_tuning.jobs.create(
        training_file=response.id,  # Use the ID from the uploaded file
        model="gpt-4o-mini"
    )
    print(f"Fine-tuning job created: {job.id}")
except Exception as e:
    print(f"Error creating fine-tuning job: {e}")