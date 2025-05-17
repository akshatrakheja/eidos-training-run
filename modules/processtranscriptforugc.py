import json
from transformers import pipeline

# Load Hugging Face Emotion Model
emotion_model = pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student")

def classify_emotion(text):
    """Classify the emotional tone of a given text."""
    result = emotion_model(text)
    return result[0]["label"]

def score_intensity(text):
    """Estimate the intensity of a message based on its length."""
    return min(len(text.split()) / 50, 1.0)  # Length-based heuristic for intensity

def preprocess_transcripts_from_jsonl(jsonl_file):
    """
    Read transcripts from a JSONL file and preprocess them.

    Args:
        jsonl_file (str): Path to the JSONL file containing transcripts.

    Returns:
        list: Preprocessed conversations in a structured format.
    """
    conversations = []
    conversation_id = 0
    
    # Open and read the JSONL file
    with open(jsonl_file, "r") as file:
        for line in file:
            transcript = json.loads(line.strip())  # Each line is a JSON object
            conversation = {"conversation_id": conversation_id, "messages": []}
            
            for exchange in transcript["messages"]:
                role = exchange["role"]
                content = exchange["content"]
                
                # Initialize metadata
                metadata = {}
                if role == "user":
                    metadata["emotion"] = classify_emotion(content)
                    metadata["intensity"] = score_intensity(content)
                    metadata["tone"] = "context-dependent"
                elif role == "assistant":
                    metadata["intent"] = exchange.get("intent", "explore")
                    metadata["tone"] = exchange.get("tone", "neutral")
                    metadata["weight"] = exchange.get("weight", 1)
                
                # Append message with metadata
                conversation["messages"].append({"role": role, "content": content, "metadata": metadata})
            
            conversations.append(conversation)
            conversation_id += 1

    return conversations

# Example usage
if __name__ == "__main__":
    # Path to the JSONL file containing transcripts
    jsonl_file_path = "../weightedjsonl/cleaned_transcripts.jsonl"
    
    # Preprocess transcripts from the JSONL file
    preprocessed_conversations = preprocess_transcripts_from_jsonl(jsonl_file_path)
    
    # Save the preprocessed conversations to a JSON file
    output_file = "preprocessed_conversations.json"
    with open(output_file, "w") as f:
        json.dump(preprocessed_conversations, f, indent=4)
    
    print(f"Preprocessed conversations saved to {output_file}")