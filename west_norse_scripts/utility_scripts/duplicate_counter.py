from collections import Counter
import json

# Load entities
print(f"\nLoading entities from {"dipl_entities.json"}...")
with open("dipl_entities.json", 'r', encoding='utf-8') as file:
    entities = json.load(file)

# Check for duplicate entities
entity_texts = Counter()
duplicates = set()

for entity in entities:
    text = entity['dipl_text']
    entity_texts[text] += 1
    if entity_texts[text] > 1:
        duplicates.add(text)

if duplicates:
    print("\nWARNING: Duplicate entities found in input file:")
    for text in sorted(duplicates):
        count = entity_texts[text]
        print(f"'{text}' appears {count} times")
else:
    print("\nNo duplicate entities found in input file")
