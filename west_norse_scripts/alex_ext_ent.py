import re
import json
from bs4 import BeautifulSoup

def extract_entities(xml_text):
    """Extracts entities from the XML text based on name type."""
    entities = []

    # First process all <name> elements
    name_pattern = re.compile(r'<name[^>]*type="([^"]*)"[^>]*>(.*?)</name>', re.DOTALL)
    for name_match in name_pattern.finditer(xml_text):
        name_type = name_match.group(1).lower()
        name_content = name_match.group(2)

        # Extract all <w> elements within this name
        w_pattern = re.compile(r'<w[^>]*>(.*?)</w>', re.DOTALL)
        dipl_parts = []
        norm_parts = []

        for w_match in w_pattern.finditer(name_content):
            w_content = w_match.group(1)

            # Extract diplomatic text
            dipl_match = re.search(r'<me:dipl>(.*?)</me:dipl>', w_content, re.DOTALL)
            if dipl_match:
                dipl_text = re.sub(r'<[^>]+>', '', dipl_match.group(1))
                dipl_parts.append(dipl_text.strip())

            # Extract normalized text
            norm_match = re.search(r'<me:norm>(.*?)</me:norm>', w_content, re.DOTALL)
            if norm_match:
                norm_text = norm_match.group(1).strip()
                norm_parts.append(norm_text)

        if dipl_parts or norm_parts:
            dipl_text = ' '.join(dipl_parts)
            norm_text = ' '.join(norm_parts)

            # Map name types to labels and subtypes
            type_mapping = {
                'person': {'label': 'PER'},
                'place': {'label': 'LOC'},
                'ship': {'label': 'SHIP'},
                'mountain': {'label': 'LOC', 'subtype': 'MTN'},
                'river': {'label': 'LOC', 'subtype': 'RIV'},
                'region': {'label': 'LOC', 'subtype': 'REG'},
                'settlement': {'label': 'LOC', 'subtype': 'SET'},
                'country': {'label': 'LOC', 'subtype': 'CNY'},
            }

            mapping = type_mapping.get(name_type, {'label': 'PER'})
            entity = {
                'dipl_text': dipl_text,
                'norm_text': norm_text,
                'label': mapping['label']
            }

            if 'subtype' in mapping:
                entity['subtype'] = mapping['subtype']

            entities.append(entity)

    return entities

def main():
    # Read the original text from the file
    input_file = 'AM-519-a-4to_alexanders-saga.xml.txt'
    output_file = 'alexander_entities.json'

    print(f"Extracting entities from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as file:
        xml_text = file.read()

    # Extract entities
    entities = extract_entities(xml_text)

    # Save the entities to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete. Found {len(entities)} entities.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
