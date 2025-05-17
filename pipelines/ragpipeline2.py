import json
import time
import hashlib
import re
import os
from dotenv import load_dotenv
from pydantic import BaseModel
# from numpy import dot
import numpy as np
from numpy.linalg import norm
import random
from pydub import AudioSegment
from pydub.playback import play
import os
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import subprocess
import threading
import queue
import openai

# Load environment variables
load_dotenv()

# Constants
INDEX_NAME = os.getenv("INDEX_NAME", "eidos-data")
NEWSPIECE = "What does a post-AGI world look like?"
DIMENSION = int(os.getenv("DIMENSION", "1536"))
METRIC = os.getenv("METRIC", "cosine")
NAMESPACE_SENTENCES = os.getenv("NAMESPACE_SENTENCES", "sentences")
NAMESPACE_CONVERSATIONS = os.getenv("NAMESPACE_CONVERSATIONS", "conversations")
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "newconversations")
# change the function
USER_INPUT_SEARCH_FUNCTION='query_sentences'
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

class GetExpertQuote(BaseModel):
    pass  # Empty class definition for now

tools = [
  {
      "type": "function",
      "function": {
          "name": "query_sentences",
          "description": "Find relevant responses from other users. Call this when the user expresses an opinion and having other users' views would be useful.",
          "parameters": {
              "type": "object",
              "properties": {
                  "query": {"type": "string"},
                #   "session_set"
              },
          },
      },
  },
  openai.pydantic_function_tool(GetExpertQuote)
]


# key is called OPENAI_API_KEY and PINECONE_KEY

# OpenAI and Pinecone setup
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

if not pinecone_client.has_index(INDEX_NAME):
    pinecone_client.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# while not pinecone_client.describe_index(INDEX_NAME).status["ready"]:
#     time.sleep(1)
MAX_WAIT_TIME = 60  # seconds
start_time = time.time()

while not pinecone_client.describe_index(INDEX_NAME).status["ready"]:
    if time.time() - start_time > MAX_WAIT_TIME:
        raise TimeoutError(f"Pinecone index '{INDEX_NAME}' did not become ready in {MAX_WAIT_TIME} seconds.")
    time.sleep(1)

index = pinecone_client.Index(INDEX_NAME)

# Embedding Function
def get_embedding(text: str):
    print(f"HERE IS THE EMBEDDING: {text}")
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# # Cosine Similarity Function
# def cosine_similarity(vec1, vec2):
#     return dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def cosine_similarity(vec1, vec2):
    numerator = np.matmul(vec1, vec2)
    denominator = (norm(vec1) * norm(vec2))
    return numerator / denominator if denominator != 0 else 0.0


# Tone and Intensity Analysis Function
def analyze_tone_intensity(sentence):
    try:
        prompt = f"""
        Analyze the tone and intensity of the following sentence:

        "{sentence}"

        Provide:
        1. Tone: (e.g., curious, confident, skeptical, empathetic, frustrated, etc.)
        2. Intensity: (scale 0 to 1, where 1 is very intense and 0 is not intense at all).
        Format your response as: "Tone: <tone>, Intensity: <intensity>"
        """
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        analysis = response.choices[0].message.content
        tone, intensity = analysis.split(", ")
        tone = tone.split(": ")[1].strip()
        intensity = float(intensity.split(": ")[1].strip())
        return tone, intensity
    except Exception as e:
        print(f"Error analyzing tone/intensity: {e}")
        return "neutral", 0.0

# Assign Names Based on Conversation ID
def assign_name(conversation_id, name_map):
    if conversation_id not in name_map:
        name_map[conversation_id] = random.choice(NAME_POOL)
    return name_map[conversation_id]

def save_session_to_jsonl(session, file_path):
    with open(file_path, "w") as f:
        json.dump(session, f)
        f.write("\n")

# Index Session Data in Pinecone with Metadata
def index_session_data(session, conversation_id, session_set):
    vectors = []
    name_map = {}  # Map conversation_id to assigned names

    for i, message in enumerate(session["messages"]):
        if message["role"] == "user":  # Only index user messages
            sentences = re.split(r'(?<=[.!?]) +', message["content"])
            for j, sentence in enumerate(sentences):
                if len(sentence.split()) <= 3:  # Skip short sentences
                    continue

                session_set.add(sentence)  # Track session data
                tone, intensity = analyze_tone_intensity(sentence)
                name = assign_name(conversation_id, name_map)

                unique_id = f"{conversation_id}-user-{i}-{j}-{hashlib.sha256(sentence.encode()).hexdigest()[:8]}"
                embedding = get_embedding(sentence)
                if embedding is None:
                    continue

                metadata = {
                    "conversation_id": conversation_id,
                    "role": "user",
                    "sentence": sentence,
                    "tone": tone,
                    "intensity": intensity,
                    "user_name": name,
                }
                vectors.append((unique_id, embedding, metadata))

    # Upsert in Pinecone
    if vectors:
        index.upsert(vectors=vectors, namespace=NAMESPACE_SENTENCES)

# Query Pinecone for Sentences with Contextual Matching
def query_sentences(query, session_set, top_k=3, block_k=3):
    # print(query)
    query_vector = get_embedding(query)
    if query_vector is None:
        print("Error: Unable to generate query embedding.")
        return []

    try:
        # Query Pinecone for text blocks
        results = index.query(
            vector=query_vector,
            top_k=block_k,
            namespace=NAMESPACE_SENTENCES,
            include_metadata=True
        )

        matched = ""
        response = ""

        if "matches" in results and results["matches"]:
            for match in results["matches"]:
                metadata = match["metadata"]
                sentence = metadata.get("sentence", "")
                name = metadata.get("user_name", "")
                
                if sentence in session_set:
                    continue
                
                return sentence
            
        return -1
    except Exception as e:
        print(f"Error querying sentences: {e}")
        return []

def call_llm(model: str, conversation_history):
    # prompt = f"Generate a follow-up question based on this context: \"{context}\"."
    try:
        if (model == "gpt-4o"):
            response = openai_client.chat.completions.create(
                model=model,
                messages=conversation_history,
                tools=tools)
        else :
            response = openai_client.chat.completions.create(
                model=model,
                messages=conversation_history,
            )
        return response.choices[0].message
    except Exception as e:
        print(f"Error generating question: {e}")
        return "Can you elaborate further?"
    

# Worker thread for sequential audio playback
def audio_worker():
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:  # Sentinel to terminate the thread
            break
        try:
            # Play the audio file using ffplay
            process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", audio_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            process.wait()  # Wait for the current audio to finish
        except Exception as e:
            print(f"Error during audio playback: {e}")
        finally:
            # Remove the temporary audio file after playback
            if os.path.exists(audio_file):
                os.remove(audio_file)
                # print(f"Temporary audio file {audio_file} removed.")
        audio_queue.task_done()

# Start the audio worker thread
# audio_thread = threading.Thread(target=audio_worker, daemon=True)
# audio_thread.start()

# Function to generate and queue audio for playback
def playback_sentence_with_queue(sentence, output_file="output.mp3"):
    try:
        # print(f"Generating TTS for sentence: {sentence}")

        # Generate TTS using OpenAI
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=sentence,
        )

        # Save audio to a temporary file
        response.stream_to_file("output.mp3")
        # print(f"Audio saved to {output_file}")

        # Add the audio file to the queue for sequential playback
        audio_queue.put(output_file)

    except Exception as e:
        print(f"Error: {e}")

# Cleanup function to stop the worker thread
# def stop_audio_worker():
#     audio_queue.put(None)  # Add sentinel to signal thread termination
#     audio_thread.join()

# Continuous Chat Loop
def run_chat(finetuned_model):
    conversation_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]
    print("System: Let's get started! (Type 'exit' to end the session.)")
    # system_prompt = "Have you ever had a near-death experience?"
    # print(system_prompt)
    session = {"messages": []}
    # first_msg = call_llm(finetuned_model, None)
    conversation_history = [
        {"role": "system", "content": f"You are a journalist. Your job is to get the user to say interesting things. Act like a jouralist/news operator interviewing the user. You're exploring the question {NEWSPIECE}. Your job is to ask users questions and engage them. Add to the discussion by succintly giving other users' responses by calling the {USER_INPUT_SEARCH_FUNCTION}. If the user says something that is verifiably infactual, then challenge them."}
    ]
    first_msg = call_llm(finetuned_model, conversation_history)
    conversation_history.append({"role": "assistant", "content": first_msg.content})
    session_set = set()

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            save_session_to_jsonl(session, f"{OUTPUT_FOLDER}/{conversation_id}.jsonl")
            print("System: Goodbye!")
            index_session_data(session, conversation_id, session_set)
            print(f"Session saved and indexed with ID: {conversation_id}")
            break

        session["messages"].append({"role": "user", "content": user_input})
        session_set.update(re.split(r'(?<=[.!?]) +', user_input))

        conversation_history.append({"role": "user", "content": user_input})
        check = call_llm("gpt-4o", conversation_history)
        search_flag = check.function_call or check.tool_calls
        response = call_llm(finetuned_model, conversation_history)
        follow_up_question = response.content
        


        if search_flag:
            print(f"ARGUMENTS: {response.tool_calls[0].function.arguments}")
            arg = response.tool_calls[0].function.arguments
            query = json.loads(arg)["query"]

            print(f"QUERY {query}")
            sentence = query_sentences(query, session_set)
            function_call_result_message = {
                "role": "tool",
                "content": json.dumps({
                    "sentence": sentence
                }),
                "tool_call_id": response.tool_calls[0].id
            }

            cache = conversation_history
            cache[0] = {"role": "system", "content": f"You are a journalist/reporter embedded into a social media's backend. You want to take the conversation in this turn: '{query}'. Based on this, answer in 2-3 sentences by contextualizing, and challenging or supporting the user through something interesting in response to the last user input. Do it in a coherent way and weave only parts of this passage if you think it'll add to your argument: '{sentence}'. If you do, do it by quoting verbatim and saying 'another used said: ' anytime you do it."}
            cache.append(response)
            cache.append(function_call_result_message)
            # Call the OpenAI API's chat completions endpoint to send the tool call result back to the model

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=cache
            )
            print(f"System: {response.choices[0].message.content}")

            session["messages"].append({"role": "assistant", "content": response.choices[0].message.content})
            conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
        
        else:
            print(f"System: {follow_up_question}")
            session["messages"].append({"role": "assistant", "content": follow_up_question})
            conversation_history.append({"role": "assistant", "content": follow_up_question})

        # Prompt user for scoring
        while True:
            try:
                score = int(input("Score the assistant's response (1 or 0): "))
                if score in [0, 1]:
                    break
                else:
                    print("Please enter 1 or 0.")
            except ValueError:
                print("Invalid input. Please enter 1 or 0.")

        # Store the score in the session for RLHF fine-tuning
        session["messages"][-1]["weight"] = score
        # user_input = input("You: ")
        # if user_input.lower() == "exit":
        #     save_session_to_jsonl(session, f"{OUTPUT_FOLDER}/{conversation_id}.jsonl")
        #     print("System: Goodbye!")
        #     index_session_data(session, conversation_id, session_set)  # Upload data at end of session
        #     print(f"Session saved and indexed with ID: {conversation_id}")
        #     break

        # session["messages"].append({"role": "user", "content": user_input})
        # session_set.update(re.split(r'(?<=[.!?]) +', user_input))

        # conversation_history.append({"role": "user", "content": user_input})
        # follow_up_question = generate_question(finetuned_model, conversation_history)
        # playback_sentence_with_queue(follow_up_question)
        
        # target_tone, target_intensity = analyze_tone_intensity(follow_up_question)
        # if (follow_up_question not in session_set):
        #     sentence = query_sentences(follow_up_question, session_set, target_tone=target_tone)
        # # print(len(sentences))
        # print(sentence)
        # print(f"System: {follow_up_question}")

        # # playback_sentence_with_queue(sentence)
        
        # # score = input("Score this question: ")
        # session["messages"].append({"role": "assistant", "content": follow_up_question})
        # conversation_history.append({"role": "assistant", "content": follow_up_question})





        # # Prompt for scoring the question
        # while True:
        #     try:
        #         score = int(input("Score: "))
        #         if score in [0, 1]:
        #             break
        #         else:
        #             print("Please enter 1 or 0.")
        #     except ValueError:
        #         print("Invalid input. Please enter 1 or 0.")

        # # Update weight for the assistant response
        # session["messages"][-1]["weight"] = score







        # if sentences:

        #     print("Here's what others have said:")
        #     for s in sentences:
        #         print(f"{s['user_name']} said - {s['sentence']} (Tone: {s['tone']}, Intensity: {s['intensity']})")

        # session["messages"].append({"role": "assistant", "content": follow_up_question})


# Main Execution
if __name__ == "__main__":
    # finetuned_model = "ft:gpt-4o-mini-2024-07-18:eidos:weighted-5:AY6pwwuw"
    finetuned_model = "ft:gpt-4o-mini-2024-07-18:eidos:onweighted4:AfLWLcNw"
    # finetuned_model = "ft:gpt-4o-mini-2024-07-18:eidos:finetuningonfinetuningonweighted3:AY6knjju"
    # print("running")
    print("i am running right now!")
    run_chat(finetuned_model)
    # stop_audio_worker()