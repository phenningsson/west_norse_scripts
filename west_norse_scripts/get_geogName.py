import re
import json
from collections import defaultdict

def extract_choices(w_content):
    """Extracts norm and norm texts from a <w> content."""
    choice_match = re.search(r'<choice>(.*?)</choice>', w_content, re.DOTALL)
    if not choice_match:
        return None, None
    choice_content = choice_match.group(1)
    norm_match = re.search(r'<me:norm>(.*?)</me:norm>', choice_content, re.DOTALL)
    norm_match = re.search(r'<me:norm>(.*?)</me:norm>', choice_content, re.DOTALL)
    norm_text = norm_match.group(1).strip() if norm_match else ""
    norm_text = norm_match.group(1).strip() if norm_match else ""
    # Clean up norm_text by removing XML tags
    norm_text = re.sub(r'<[^>]+>', '', norm_text)
    return norm_text, norm_text

def extract_entity_texts(entity_content):
    """Extracts norm and norm texts from an entity's content."""
    w_pattern = re.compile(r'<w[^>]*>(.*?)</w>', re.DOTALL)
    w_matches = w_pattern.finditer(entity_content)
    norm_parts = []
    norm_parts = []
    for w_match in w_matches:
        w_content = w_match.group(1)
        norm, norm = extract_choices(w_content)
        if norm:
            norm_parts.append(norm)
        if norm:
            norm_parts.append(norm)
    norm_text = ' '.join(norm_parts)
    norm_text = ' '.join(norm_parts)
    return norm_text, norm_text

def extract_geog_names(xml_text):
    """Extracts geogName entities from XML text."""
    # Pattern for geogName with optional type attribute
    geogname_pattern = re.compile(r'<geogName[^>]*type="([^"]*)"[^>]*>(.*?)</geogName>', re.DOTALL)

    entities = []

    for match in geogname_pattern.finditer(xml_text):
        geog_type = match.group(1).lower()
        entity_content = match.group(2)
        norm_text, norm_text = extract_entity_texts(entity_content)

        if norm_text:  # Only add if we have normomatic text
            # Map geogName types to subtypes (default to OTR if not specified)
            subtype_map = {
                'mountain': 'MTN',
                'river': 'RIV',
                'region': 'REG',
                'settlement': 'SET',
                'country': 'CNY',
                # Add more mappings as needed
            }

            subtype = subtype_map.get(geog_type, 'OTR')  # Default to OTR for unknown types

            entities.append({
                'norm_text': norm_text,
                'label': 'LOC',
                'subtype': subtype
            })

    return entities

def main():
    # Read the XML file
    input_file = 'menota_normalised/islendingabok/islendingabok.xml.txt'
    output_file = 'menota_normalised/islendingabok/geogname_entities.json'

    print(f"Extracting geogName entities from {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as file:
        xml_text = file.read()

    # Extract entities
    entities = extract_geog_names(xml_text)

    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete. Found {len(entities)} geogName entities.")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
