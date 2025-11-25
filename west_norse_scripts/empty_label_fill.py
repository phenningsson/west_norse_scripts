#!/usr/bin/env python3
"""
Convert empty labels to PER and split compound labels into label + subtype.
- Empty label → "PER"
- "PER-ROL" → "label": "PER", "subtype": "ROL"
"""

import json


def process_entities(input_file, output_file):
    """Process labels: fill empty ones and split compound labels."""
    
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    # Track changes
    filled_count = 0
    split_count = 0
    
    # Process each entity
    for entity in entities:
        label = entity['label']
        
        # Case 1: Empty label → PER
        if label == '':
            entity['label'] = 'PER'
            filled_count += 1
        
        # Case 2: Compound label (e.g., "PER-ROL") → split into label + subtype
        elif '-' in label:
            parts = label.split('-', 1)  # Split on first hyphen only
            entity['label'] = parts[0]
            entity['subtype'] = parts[1]
            split_count += 1
    
    # Save the updated JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    return filled_count, split_count, len(entities)


def main():
    input_file = "ihpc/1275_morkin/1275_morkin_entities_init.json"
    output_file = "ihpc/1275_morkin/1275_morkin_entities.json"
    
    print("Processing entity labels...")
    print("  - Converting empty labels to PER")
    print("  - Splitting compound labels (e.g., PER-ROL → PER + subtype: ROL)")
    
    filled, split, total = process_entities(input_file, output_file)
    
    print(f"\n✓ Filled {filled} empty labels with PER")
    print(f"✓ Split {split} compound labels into label + subtype")
    print(f"✓ Total entities: {total}")
    print(f"\n✓ Saved to: {output_file}")


if __name__ == "__main__":
    main()