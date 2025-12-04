import json

def create_ner_json_dataset(text_file, entity_positions_file, word_boundaries_file, output_file, id, name):
    """
    Creates a NER dataset in JSON format.
    Args:
        text_file: Path to the clean text file
        entity_positions_file: Path to JSON file with entity positions
        word_boundaries_file: Path to JSON file with word boundaries
        output_file: Path to save the JSON file
        id: Integer ID for the document
        name: Name of the work/manuscript
    """
    # Load clean text and replace newlines with spaces
    with open(text_file, 'r', encoding='utf-8') as f:
        clean_text = f.read().replace('\n', ' ')

    # Load entity positions
    with open(entity_positions_file, 'r', encoding='utf-8') as f:
        entity_positions = json.load(f)

    # Load word boundaries
    with open(word_boundaries_file, 'r', encoding='utf-8') as f:
        word_boundaries = json.load(f)

    # Extract word boundary arrays
    word_start_chars = [word['start'] for word in word_boundaries]
    word_end_chars = [word['end'] for word in word_boundaries]

    # Sort entities by start position
    sorted_entities = sorted(entity_positions, key=lambda x: x['start'])

    # Extract entity information
    entity_types = []
    entity_start_chars = []
    entity_end_chars = []

    for entity in sorted_entities:
        # Combine label and subtype if subtype exists
        if 'subtype' in entity and entity['subtype']:
            combined_label = f"{entity['label']}-{entity['subtype']}"
        else:
            combined_label = entity['label']

        entity_types.append(combined_label)
        entity_start_chars.append(entity['start'])
        entity_end_chars.append(entity['end'])

    # Create the data structure (flat, not nested in arrays)
    data = {
        "text": clean_text,
        "entity_types": entity_types,
        "entity_start_chars": entity_start_chars,
        "entity_end_chars": entity_end_chars,
        "word_start_chars": word_start_chars,
        "word_end_chars": word_end_chars,
        "id": id,
        "work": name
    }

    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"JSON dataset saved to {output_file}")
    return data


def main():
    # File paths
    text_file = '/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/clean_texts/adjudications.txt'
    entity_positions_file = '/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/binder_entities/adjudications_entities_positions.json'
    word_boundaries_file = '/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/binder_entities/adjudications_word_positions.json'
    output_file = '/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/adjudications_mim_ner_dataset_dataset.json'

    # Document metadata
    id = 16
    name = "Adjudications"

    # Create and save the dataset
    data = create_ner_json_dataset(
        text_file, 
        entity_positions_file, 
        word_boundaries_file, 
        output_file,
        id,
        name
    )

    # Print summary
    print(f"\nDataset summary:")
    print(f"  Text length: {len(data['text'])} characters")
    print(f"  Entities: {len(data['entity_types'])}")
    print(f"  Words: {len(data['word_start_chars'])}")


if __name__ == "__main__":
    main()