#!/usr/bin/env python3
"""
Script to remove the 'subtype' field from JSON entries.
"""

import json

# ============================================
# SET YOUR FILE NAMES HERE
# ============================================
INPUT_FILE = "menota/olafs_saga_helga/dipl_olaf_entities_flat.json"
OUTPUT_FILE = "menota/olafs_saga_helga/dipl_olaf_entities_flattened.json"
# ============================================

def remove_subtype(input_file, output_file):
    """
    Remove 'subtype' field from all entries in the JSON file.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
    """
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Remove 'subtype' field from each entry
    cleaned_data = []
    for entry in data:
        # Create a new dict without the 'subtype' key
        cleaned_entry = {k: v for k, v in entry.items() if k != 'subtype'}
        cleaned_data.append(cleaned_entry)
    
    # Save the cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Removed 'subtype' field from {len(cleaned_data)} entries")
    print(f"✓ Saved to: {output_file}")

if __name__ == "__main__":
    remove_subtype(INPUT_FILE, OUTPUT_FILE)