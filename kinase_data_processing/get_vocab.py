import json

# File paths
input_file = "mapping.jsonl"
output_file = "vocabulary.txt"

# Initialize an empty set to store unique fragments
unique_fragments = set()

# Read the JSONL file line by line
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        data = json.loads(line.strip())  # Load the JSON object
        for value in data.values():  # Extract the values
            fragments = value.split('.')  # Split the string by '.'
            unique_fragments.update(fragments)  # Add to set

# Write the unique fragments to a file, one per line
with open(output_file, "w", encoding="utf-8") as file:
    for fragment in sorted(unique_fragments):  # Sort for consistency
        file.write(fragment + "\n")

print(f"Saved {len(unique_fragments)} unique fragments to {output_file}")
