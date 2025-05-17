import json
import re

def fix_roles_and_split(input_file, output_file):
    """
    Reads a JSONL file, fixes user/assistant role errors,
    and splits messages with mixed roles into separate messages.
    """
    corrected_data = []

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():  # Skip empty lines
                continue
            try:
                # Parse the JSON object
                conversation = json.loads(line)
                messages = conversation.get("messages", [])

                # Iterate through messages and fix role issues
                fixed_messages = []
                for message in messages:
                    role = message.get("role", "").lower()
                    content = message.get("content", "")

                    # Check if the content includes "This is user" or "This is system"
                    if "this is user" in content.lower() or "this is system" in content.lower():
                        # Split content into separate messages based on "This is" phrases
                        parts = re.split(r"(this is user|this is system)", content, flags=re.IGNORECASE)
                        current_role = role
                        for part in parts:
                            part = part.strip()
                            if not part:
                                continue
                            if part.lower() == "this is user":
                                current_role = "user"
                            elif part.lower() == "this is system":
                                current_role = "assistant"
                            else:
                                # Add the split message with the determined role
                                new_message = {"role": current_role, "content": part.strip()}
                                if current_role == "assistant":
                                    new_message["weight"] = 1  # Add weight for assistant
                                fixed_messages.append(new_message)
                    else:
                        # Add the original message if no split is needed
                        fixed_message = {"role": role, "content": content}
                        if role == "assistant":
                            fixed_message["weight"] = message.get("weight", 1)
                        fixed_messages.append(fixed_message)

                # Append the corrected conversation
                corrected_data.append({"messages": fixed_messages})

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON line: {line}")
                continue

    # Write the corrected data back to a new JSONL file
    with open(output_file, "w", encoding="utf-8") as outfile:
        for conversation in corrected_data:
            json.dump(conversation, outfile)
            outfile.write("\n")

    print(f"Role errors corrected and saved to {output_file}")

# Example Usage
input_file = "/Users/rakheja/Documents/conversation-analysis/weightedjsonl/corrected_newround.jsonl"
output_file = "/Users/rakheja/Documents/conversation-analysis/weightedjsonl/final_corrected_newround.jsonl"

fix_roles_and_split(input_file, output_file)