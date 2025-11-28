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
    """Extracts entities from the XML text with proper location subtypes and persName handling."""
    # Location subtype mapping
    location_subtypes = {
        'region': 'REG',
        'settlement': 'SET',
        'country': 'CNY'
    }

    # Pattern for addName with subtype
    addname_pattern = re.compile(r'<addName[^>]*type="([^"]*)"[^>]*>(.*?)</addName>', re.DOTALL)

    # Pattern for persName
    persname_pattern = re.compile(r'<persName>(.*?)</persName>', re.DOTALL)

    # Pattern for placeName
    placename_pattern = re.compile(r'<placeName>(.*?)</placeName>', re.DOTALL)

    entities = []

    # Process persName entities
    for persname_match in persname_pattern.finditer(original_text):
        persname_content = persname_match.group(1)

        # Check for forename, roleName, and addName within persName
        forename_match = re.search(r'<forename>(.*?)</forename>', persname_content, re.DOTALL)
        rolename_match = re.search(r'<roleName[^>]*>(.*?)</roleName>', persname_content, re.DOTALL)
        addname_match = re.search(r'<addName[^>]*type="([^"]*)"[^>]*>(.*?)</addName>', persname_content, re.DOTALL)

        # Extract forename if exists
        if forename_match:
            forename_content = forename_match.group(1)
            forename_dipl, forename_norm = extract_entity_texts(forename_content)
            if forename_dipl or forename_norm:
                # Add forename as PER entity
                entities.append({
                    'dipl_text': forename_dipl,
                    'norm_text': forename_norm,
                    'label': 'PER'
                })

                # Check for roleName and combine with forename if exists
                if rolename_match:
                    rolename_content = rolename_match.group(1)
                    rolename_dipl, rolename_norm = extract_entity_texts(rolename_content)
                    combined_dipl = f"{forename_dipl} {rolename_dipl}".strip()
                    combined_norm = f"{forename_norm} {rolename_norm}".strip()
                    if combined_dipl or combined_norm:
                        entities.append({
                            'dipl_text': combined_dipl,
                            'norm_text': combined_norm,
                            'label': 'PER',
                            'subtype': 'ROL'
                        })

                # Check for addName and combine with forename if exists
                if addname_match:
                    addname_type = addname_match.group(1).lower()
                    addname_content = addname_match.group(2)
                    addname_dipl, addname_norm = extract_entity_texts(addname_content)

                    # Get the subtype abbreviation
                    subtype_map = {
                        'patronym': 'PAT',
                        'patronymic': 'PAT',
                        'nickname': 'NIC',
                    }
                    subtype = subtype_map.get(addname_type, addname_type.upper())

                    combined_dipl = f"{forename_dipl} {addname_dipl}".strip()
                    combined_norm = f"{forename_norm} {addname_norm}".strip()
                    if combined_dipl or combined_norm:
                        entities.append({
                            'dipl_text': combined_dipl,
                            'norm_text': combined_norm,
                            'label': 'PER',
                            'subtype': subtype
                        })

    # Process standalone addName entities (not within persName)
    for addname_match in addname_pattern.finditer(original_text):
        # Only process if not already handled within persName
        if not re.search(r'<persName>.*?</persName>', addname_match.group(0), re.DOTALL):
            subtype = addname_match.group(1).lower()
            entity_content = addname_match.group(2)
            dipl_text, norm_text = extract_entity_texts(entity_content)
            if dipl_text or norm_text:
                entity = {
                    'dipl_text': dipl_text,
                    'norm_text': norm_text,
                    'label': 'PER',
                }
                subtype_map = {
                    'patronym': 'PAT',
                    'patronymic': 'PAT',
                    'nickname': 'NIC',
                }
                entity['subtype'] = subtype_map.get(subtype, subtype.upper())
                entities.append(entity)

    # Process placeName entities and their children
    for placename_match in placename_pattern.finditer(original_text):
        placename_content = placename_match.group(1)
        # Check for location children (region, settlement, country)
        for child_tag in location_subtypes.keys():
            child_pattern = re.compile(f'<{child_tag}>(.*?)</{child_tag}>', re.DOTALL)
            for child_match in child_pattern.finditer(placename_content):
                child_content = child_match.group(1)
                dipl_text, norm_text = extract_entity_texts(child_content)
                if dipl_text or norm_text:
                    entities.append({
                        'dipl_text': dipl_text,
                        'norm_text': norm_text,
                        'label': 'LOC',
                        'subtype': location_subtypes[child_tag]
                    })

    return entities

# Read the original text from the xml file
with open('/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/islendingabok/islendingabok.xml.txt', 'r', encoding='utf-8') as file:
    original_text = file.read()

# Extract entities
entities = extract_entities(original_text)

# Save the entities to a JSON file
with open('/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/islendingabok/islendingabok_entities.json', 'w', encoding='utf-8') as f:
    json.dump(entities, f, ensure_ascii=False, indent=2)

print("Extraction complete.")
