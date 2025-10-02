import json

def simplify_entities(input_file, output_file):
    """
    Creates a simplified version of the entities file without norm_text fields.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    simplified = []
    for entity in entities:
        # Create a simplified version without norm_text
        simplified_entity = {
            'dipl_text': entity['dipl_text'],
            'label': entity['label']
        }
        # Include subtype if it exists
        if 'subtype' in entity:
            simplified_entity['subtype'] = entity['subtype']
        simplified.append(simplified_entity)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, ensure_ascii=False, indent=2)

    print(f"Simplified entities saved to {output_file}")

if __name__ == "__main__":
    input_file = 'alexander_entities.json'
    output_file = 'alexander_entities_dipl.json'
    simplify_entities(input_file, output_file)
