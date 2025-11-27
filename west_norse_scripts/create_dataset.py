import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def create_ner_dataset(clean_text, entity_positions, word_boundaries_file, output_file):
    """
    Creates a NER dataset and saves as Parquet file.
    Args:
        clean_text: The full clean text
        entity_positions: List of entities with positions
        word_boundaries_file: Path to JSON file with word boundaries
        output_file: Path to save the Parquet file
    """
    # Load word boundaries from JSON file
    with open(word_boundaries_file, 'r', encoding='utf-8') as f:
        word_boundaries = json.load(f)
    word_starts = [word['start'] for word in word_boundaries]
    word_ends = [word['end'] for word in word_boundaries]

    # Create and populate structure
    data = {
        "text": [clean_text],
        "entity_types": [[]],  
        "entity_start_chars": [[]],
        "entity_end_chars": [[]],
        "word_start_chars": [word_starts],
        "word_end_chars": [word_ends],
        "id": [8],
        "year": ["1350"],  # Year of manuscript
        "lang": ["isl"],  # Language code (three letters)
        "work": ["Codex Wormianus: AM 242 fol"],  # Name of the work
        "folio": ["1-169"], # From which folio(s) of the manuscripts the current text comes from
        "text_rep_level": ["dipl"],  # dipl for diplomatic; norm for normalised; facs for facsimili
        "main_editor": ["Karl Gunnar Johansson"],  # Update with actual editor name
        "annotator": ["null"]  # Update with annotator name
    }

    # Sort entities by their start position for consistent ordering
    sorted_entities = sorted(entity_positions, key=lambda x: x['start'])

    # Extract entity information with combined label-subtype
    entity_types = []
    entity_start_chars = []
    entity_end_chars = []

    for entity in sorted_entities:
        # Formats the label as "MAIN-SUBTYPE" if subtype exists, otherwise just "MAIN"
        if 'subtype' in entity and entity['subtype']:
            combined_label = f"{entity['label']}-{entity['subtype']}"
        else:
            combined_label = entity['label']

        entity_types.append(combined_label)
        entity_start_chars.append(entity['start'])
        entity_end_chars.append(entity['end'])

    # Update the entity information in data structure
    data["entity_types"][0] = entity_types
    data["entity_start_chars"][0] = entity_start_chars
    data["entity_end_chars"][0] = entity_end_chars

    # Create a pandas DataFrame
    df = pd.DataFrame(data)

    # Convert to PyArrow Table for Parquet writing
    table = pa.Table.from_pandas(df)

    # Write to Parquet file
    pq.write_table(table, output_file)
    print(f"Dataset saved to {output_file}")
    return df  # Also return the DataFrame for inspection

def main():
    # File paths
    text_file = 'flat_ner/menota/wormianus/wormianus_final_text.txt'
    entity_positions_file = 'flat_ner/menota/wormianus/dipl_wormianus_entities_flat_positions.json'
    word_boundaries_file = 'flat_ner/menota/wormianus/wormianus_word_positions.json'
    output_file = 'flat_ner/menota/wormianus/wormianus_flat_ner_dataset.parquet'

    print(f"Loading text from {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as file:
        clean_text = file.read()

    print(f"\nLoading entity positions from {entity_positions_file}...")
    with open(entity_positions_file, 'r', encoding='utf-8') as file:
        entity_positions = json.load(file)

    # Create and save the dataset
    df = create_ner_dataset(clean_text, entity_positions, word_boundaries_file, output_file)

    # Print the resulting DataFrame for verification
    print("\nDataset structure:")
    print(df)

if __name__ == "__main__":
    main()
