import os
import json
import re

def process_transcript_file(file_path):
    """
    Processes a single transcript file into JSONL format.
    """
    messages = []
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()  # Read file line by line
    
    for line in lines:
        line = line.strip()
        if not line:  # Skip empty lines
            continue
        
        # Match and extract "This is system" or "This is user" prefix
        system_match = re.match(r"(?i)^this is (?:the )?system\.\s*(.*)", line)
        user_match = re.match(r"(?i)^this is (?:the )?user(?: [\w\s]+)?\.\s*(.*)", line)

        if system_match:
            content = system_match.group(1).strip()
            if content:
                messages.append({"role": "assistant", "content": content, "weight": 1})
        elif user_match:
            content = user_match.group(1).strip()
            if content:
                messages.append({"role": "user", "content": content})
        else:
            # Handle unrecognized formats
            print(f"Warning: Unrecognized line format -> {line}")
    
    return {"messages": messages}

def process_all_files(folder_path, output_file):
    """
    Processes all transcript files in a folder and writes a JSONL output.
    """
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Process only .txt files
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {filename}")
            conversation = process_transcript_file(file_path)
            if conversation["messages"]:  # Only add non-empty conversations
                data.append(conversation)

    # Write all conversations to a JSONL file
    with open(output_file, "w", encoding="utf-8") as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + "\n")

    print(f"All transcripts processed and saved to {output_file}")

# Example Usage
if __name__ == "__main__":
    input_folder = "../newdata/round3"  # Folder containing transcripts
    output_file = "../weightedjsonl/newround.jsonl"  # Output JSONL file
    process_all_files(input_folder, output_file)