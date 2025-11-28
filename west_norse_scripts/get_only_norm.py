import json

def simplify_entities(input_file, output_file):
    """
    Creates a simplified version of the entities file without dipl_text fields.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    simplified = []
    for entity in entities:
        # Create a simplified version without dipl_text
        simplified_entity = {
            'norm_text': entity['norm_text'],
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
    input_file = '/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/alexanders_saga/alexander_saga_entities.json'
    output_file = '/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/alexanders_saga/alexander_saga_entities_norm.json'
    simplify_entities(input_file, output_file)
