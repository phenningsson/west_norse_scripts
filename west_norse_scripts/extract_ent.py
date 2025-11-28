import re
import json

def extract_choices(w_content):
    """Extracts dipl and norm texts from a <w> content."""
    choice_match = re.search(r'<choice>(.*?)</choice>', w_content, re.DOTALL)
    if not choice_match:
        return None, None
    choice_content = choice_match.group(1)

    dipl_match = re.search(r'<me:dipl>(.*?)</me:dipl>', choice_content, re.DOTALL)
    norm_match = re.search(r'<me:norm>(.*?)</me:norm>', choice_content, re.DOTALL)

    dipl_text = dipl_match.group(1).strip() if dipl_match else ""
    norm_text = norm_match.group(1).strip() if norm_match else ""

    # Clean up dipl_text by removing XML tags
    dipl_text = re.sub(r'<[^>]+>', '', dipl_text)
    return dipl_text, norm_text

def extract_entity_texts(entity_content):
    """Extracts dipl and norm texts from an entity's content."""
    w_pattern = re.compile(r'<w[^>]*>(.*?)</w>', re.DOTALL)
    w_matches = w_pattern.finditer(entity_content)

    dipl_parts = []
    norm_parts = []

    for w_match in w_matches:
        w_content = w_match.group(1)
        dipl, norm = extract_choices(w_content)
        if dipl:
            dipl_parts.append(dipl)
        if norm:
            norm_parts.append(norm)

    dipl_text = ' '.join(dipl_parts)
    norm_text = ' '.join(norm_parts)

    return dipl_text, norm_text

def extract_entities(original_text):
    """Extracts entities from the XML text."""
    # Patterns for different entity types
    patterns = {
        'placeName': (r'<placeName>(.*?)</placeName>', 'LOC', None),
        'persName': (r'<persName>(.*?)</persName>', 'PER', None),
        'addName': (r'<addName[^>]*type="([^"]*)"[^>]*>(.*?)</addName>', 'PER', 'subtype')
        
    }

    entities = []

    for tag, (pattern, main_label, subtype_attr) in patterns.items():
        compiled_pattern = re.compile(pattern, re.DOTALL)

        for match in compiled_pattern.finditer(original_text):
            if subtype_attr:  # For tags with attributes
                subtype = match.group(1)
                entity_content = match.group(2)
            else:  # For simple tags
                subtype = None
                entity_content = match.group(1)

            dipl_text, norm_text = extract_entity_texts(entity_content)
            if dipl_text or norm_text:  # At least one has content
                entity = {
                    'dipl_text': dipl_text,
                    'norm_text': norm_text,
                    'label': main_label,
                }

                # Add subtype if available
                if subtype:
                    # Map subtype to abbreviation
                    subtype_map = {
                        'patronym': 'PAT',
                        'patronymic': 'PAT',
                        'nickname': 'NIC',
                        'country': 'CNY',
                        'region': 'REG',
                        'settlement': 'SET',
                    }
                    entity['subtype'] = subtype_map.get(subtype.lower(), subtype.upper())

                entities.append(entity)

    return entities

# Read the original text from the file
with open('/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/am_162_B_0_fol/am_162_B_0_fol.xml.txt', 'r', encoding='utf-8') as file:
    original_text = file.read()

# Extract entities
entities = extract_entities(original_text)

# Save the entities to a JSON file
with open('/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/am_162_B_0_fol/am162b0_entities.json', 'w', encoding='utf-8') as f:
    json.dump(entities, f, ensure_ascii=False, indent=2)

print("Extraction complete.")
