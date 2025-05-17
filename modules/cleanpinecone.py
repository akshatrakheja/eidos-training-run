import time
import random
import json
import os
import hashlib
from dotenv import load_dotenv
from openai import OpenAI
import pinecone
# from transformers import pipeline
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

# Load environment variables
load_dotenv()

# OpenAI and Pinecone setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("INDEX_NAME", "eidos-data")
index = pinecone_client.Index(INDEX_NAME)

NAME_POOL =  [ "Porfirio Bourke",
"Giacinta Dwerryhouse",
"Odharnait May",
"Laci Hass",
"Jaida Harding",
" Shukriya Arnoni",
"Elma Eustis",
"Rashid Lombardi",
"Bennie Abbatantuono",
"Anneka Fleming",
"Sabrina Darrin",
"Brant Dipak",
"Bharat Satyavati",
"Gianmarco Ember",
"Brant Demetria",
" Corina Mahir",
"Affan Duha",
"Tatiana Fraser"
"Garnette Kailee",
"Gundula Denzil",
'Felix Singh', 'Tariq Yamamoto', 'Dimitri Dubois', 'Liam Mehta', 'Sana Bauer', 
'Yuki Liu', 'Mateo Khan', 'Aya Bauer', 'Amara Fernandez', 'Fatima Wang', 
'Tariq Hassan', 'Ava Mehta', 'Marco Rossi', 'Khalid Tanaka', 'Carlos Torres', 
'Priya Jensen', 'Chloe Patel', 'Sana Shah', 'Aisha Hernandez', 'Liam Leclerc', 
'Liam Yamamoto', 'Mina Garcia', 'Arjun Fernandez', 'Khalid Moreau', 'Ali Khan', 
'Carlos Castro', 'Noah Ahmed', 'Tariq Torres', "Aisha O'Connor", 'Sophia Liu', 
'Fatima Ahmed', 'Liam Pereira', 'Aisha Petrov', 'Priya Wang', 'Priya Rossi', 
'Aya Moreau', 'Carlos Alvarez', 'Ethan Nakamura', 'Oscar Singh', 'Amara Wang', 
'Ahmed Yamamoto', 'Maya Lopez', 'Noah Kowalski', 'Elif Yamamoto', 'Yuki Silva', 
'Jade Patel', 'Liam Dubois', 'Zoe Chavez', 'Niko Gonzalez', 'Yuki Pereira'  ]

# Helper functions
def analyze_tone_intensity(sentence):
    try:
        prompt = f"""
        Analyze the tone and intensity of the following sentence: 

        "{sentence}"

        Provide:
        1. Tone: (e.g., curious, confident, skeptical, empathetic, sarcastic, frustrated, supportive, etc.)
        2. Intensity: (scale 0 to 1, where 1 is very intense and 0 is not intense at all).
        Format your response as: "Tone: <tone>, Intensity: <intensity>". Tone should be a single word and intensity a single number
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system", "content":prompt}]
            # max_completion_tokens=50,
            # temperature=0.7
        )
        analysis = response.choices[0].message.content
        print(analysis)
        tone, intensity = analysis.split(", ")
        tone = tone.split(": ")[1].strip()
        intensity = float(intensity.split(": ")[1].strip())
        return tone, intensity
    except Exception as e:
        print(f"Error analyzing tone/intensity: {e}")
        return "neutral", 0.0

# Function to assign a name based on conversation_id
def assign_name(conversation_id, name_map):
    if conversation_id not in name_map:
        name_map[conversation_id] = random.choice(NAME_POOL)
    return name_map[conversation_id]

# Function to process messages and filter incomplete/bad sentences
def process_messages(messages, name_map):
    processed_sentences = []

    for obj in messages:
        conversation_id = obj.get("conversation_id", hashlib.sha256(str(random.random()).encode()).hexdigest()[:8])
        for message in obj["messages"]:
            if message["role"] == "user":  # Only process user messages
                sentence = message["content"].strip()
                if len(sentence.split()) <= 3:  # Filter out short/incomplete sentences
                    continue

                # Analyze tone and intensity
                tone, intensity = analyze_tone_intensity(sentence)

                # Assign or reuse a name
                user_name = assign_name(conversation_id, name_map)

                # Construct metadata
                metadata = {
                    "sentence": sentence,
                    "conversation_id": conversation_id,
                    "tone": tone,
                    "intensity": intensity,
                    "user_name": user_name
                }

                processed_sentences.append(metadata)

    return processed_sentences

# Function to upsert processed sentences to Pinecone
def upsert_to_pinecone(processed_sentences):
    vectors = []

    for sentence_data in processed_sentences:
        unique_id = hashlib.sha256(sentence_data["sentence"].encode()).hexdigest()
        # Use OpenAI embeddings for vector
        embedding = get_embedding(sentence_data["sentence"])
        if embedding is None:
            continue

        vectors.append((unique_id, embedding, sentence_data))

    # Upsert in batches
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(batch, namespace="sentences")

# Function to get embeddings
def get_embedding(text):
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for '{text}': {e}")
        return None

# Main Function
def main(input_file):
    name_map = {}
    messages = []

    # Read JSONL file
    with open(input_file, "r") as file:
        for line in file:
            messages.append(json.loads(line.strip()))

    print("Processing messages...")
    processed_sentences = process_messages(messages, name_map)

    print(f"Processed {len(processed_sentences)} sentences. Uploading to Pinecone...")
    upsert_to_pinecone(processed_sentences)

    print("Upload completed successfully.")

# Run the script
if __name__ == "__main__":
    input_file = "../rhlf/collected.jsonl"  # Replace with your JSONL file
    main(input_file)