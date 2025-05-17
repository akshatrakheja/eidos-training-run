import json
import uuid
import os
from dotenv import load_dotenv
from datetime import datetime
from openai import OpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import requests
from decimal import Decimal

# Load environment variables
load_dotenv()

# Azure Search Configurations
INDEX_NAME = os.getenv("AZURE_INDEX_NAME", "transcripts-index")
SERVICE_ENDPOINT = os.getenv("AZURE_SERVICE_ENDPOINT", "https://eidos-mvp.search.windows.net")
API_KEY = os.getenv("AZURE_API_KEY")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize Azure Search client
client = SearchClient(endpoint=SERVICE_ENDPOINT, index_name=INDEX_NAME, credential=AzureKeyCredential(API_KEY))
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Generate unique user ID
def generate_user_id():
    return str(uuid.uuid4())

def generate_relevance_to_question_score(content, question):
    prompt = [{"role": "system", "content": f"Rate the relevance of the following response to the question:\n\n{content}\n\nQuestion: {question}\n\Respond with a single number: the relevance from 0 to 1, with 2 decimal places, where 0 is not relevant and 1 is highly relevant."}]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            max_completion_tokens=10
        )
        res = response.choices[0].message.content
        return float(res)
    except Exception as e:
        print(f"Failed to generate relevance score for content: {content[:50]}..., Error: {e}")
        return ""

def generate_title(content):
    prompt = [{"role": "system", "content": f"Generate a title for the following content:\n\n{content}\n\nEnsure your answer is just the title. Make it between 2 - 15 words"}]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            max_completion_tokens=35
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to generate title for content: {content[:50]}..., Error: {e}")
        return ""    

def generate_tags(content):
    prompt = [{"role": "system", "content": f"Extract up to 5 tags each 1 word to signify what the following content is talking about:\n\n{content}\n\nEnsure your answer is just the tags in lowercase separated by commas."}]
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt,
            max_completion_tokens=50
        )
        tags = response.choices[0].message.content
        return [tag.strip() for tag in tags.split(",") if tag.strip()]  # Split and clean tags
    except Exception as e:
        print(f"Failed to generate tags for content: {content[:50]}..., Error: {e}")
        return []
    
def fetch_all_document_ids():
    search_url = f"{SERVICE_ENDPOINT}/indexes/{INDEX_NAME}/docs?api-version=2020-06-30&$select=id&$top=1000"
    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch documents: {response.status_code}, {response.text}")
    
    docs = response.json()["value"]
    return [doc["id"] for doc in docs]

# Delete documents in batches
def delete_documents(document_ids):
    delete_batch = {
        "value": [{"@search.action": "delete", "id": doc_id} for doc_id in document_ids]
    }
    delete_url = f"{SERVICE_ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version=2020-06-30"
    response = requests.post(delete_url, headers=headers, data=json.dumps(delete_batch))
    if response.status_code != 200:
        raise Exception(f"Failed to delete documents: {response.status_code}, {response.text}")
    print(f"Deleted {len(document_ids)} documents.")


# Process conversation and upload to Azure Search
def upload_conversations_to_azure(jsonl_file):
    with open(jsonl_file, "r") as f:
        i = 0
        for line in f:
            i+=1
            conversation = json.loads(line)
            # user_id = generate_user_id()
            conversation_id = str(uuid.uuid4())

            assistant_message = "How is AGI going to change your life?"
            
            # Process messages
            for message in conversation["messages"]:
                is_user = message["role"] == "user"

                response = openai_client.embeddings.create(
                    input=message["content"],
                    model="text-embedding-3-small"
                )

                print(f"generated embedding for {message['content']}")               
                # Data structure for Azure Search
                id = str(uuid.uuid4())
                document = {
                    "id": id,  # Unique document ID
                    "title": generate_title(message["content"]),
                    "conversation_id": conversation_id,
                    "role": message["role"],
                    "content": message["content"],
                    "embedding": response.data[0].embedding,  # Placeholder for embedding; will populate separately
                    "tags": generate_tags(message["content"]),
                    "audio_url": None,  # Placeholder for audio file URL
                    "newspiece": "AGI"
                }

                if is_user:
                    document["user_id"] = generate_user_id()
                    document["score"] = generate_relevance_to_question_score(message["content"], assistant_message)
                else:
                    document["user_id"] = None
                    assistant_message = message["content"]
                    document["score"] = message["weight"]

                # Upload to Azure Search
                try:
                    result = client.upload_documents(documents=[document])
                    print(f"Uploaded document ID: {document['id']}")
                except Exception as e:
                    print(f"Failed to upload document ID: {document['id']}, Error: {e}")

# Main function to run the script
if __name__ == "__main__":
    JSONL_FILE = "../weightedjsonl/cleaned_transcripts.jsonl"  # Path to your JSONL file

    document_ids = fetch_all_document_ids()
    if not document_ids:
        print("No documents to delete.")
    else:
        delete_documents(document_ids)
        print("All documents deleted successfully.")

    upload_conversations_to_azure(JSONL_FILE)
