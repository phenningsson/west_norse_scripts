#!/usr/bin/env python3
"""
Convert MIM-GOLD-NER BIO format to Binder NER JSON format.

This script extracts PERSON and LOCATION entities from the MIM-GOLD-NER corpus
(CoNLL BIO format) and converts them to the JSON format needed for Binder training.

Creates one JSON file per input TXT file.

Input format (CoNLL BIO):
    Áslaug    B-Person
    Þorgeirsdóttir    I-Person
    ...

Output format (Binder JSON):
    [
        {"dipl_text": "Áslaug Þorgeirsdóttir", "label": "PER"},
        ...
    ]
"""

import json
from pathlib import Path


# ============================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================

# Input directory containing the MIM-GOLD-NER .txt files
INPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/bio_entities"

# Output directory for the JSON files (one per input file)
OUTPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/binder_entities"

# Set to True to remove duplicate entities within each file
DEDUPLICATE = True

# ============================================================


def parse_bio_file(filepath: str) -> list[dict]:
    """
    Parse a single BIO-format file and extract Person and Location entities.
    
    Args:
        filepath: Path to the BIO format file
        
    Returns:
        List of entity dictionaries with 'dipl_text' and 'label' keys
    """
    entities = []
    current_entity_tokens = []
    current_entity_type = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Empty line signals end of sentence - flush any current entity
            if not line:
                if current_entity_tokens and current_entity_type:
                    entity_text = ' '.join(current_entity_tokens)
                    entities.append({
                        'dipl_text': entity_text,
                        'label': current_entity_type
                    })
                current_entity_tokens = []
                current_entity_type = None
                continue
            
            # Split token and tag
            parts = line.split('\t')
            if len(parts) != 2:
                # Handle malformed lines
                continue
                
            token, tag = parts
            
            # Check for B-Person or B-Location (Beginning of entity)
            if tag == 'B-Person':
                # Save any previous entity first
                if current_entity_tokens and current_entity_type:
                    entity_text = ' '.join(current_entity_tokens)
                    entities.append({
                        'dipl_text': entity_text,
                        'label': current_entity_type
                    })
                # Start new Person entity
                current_entity_tokens = [token]
                current_entity_type = 'PER'
                
            elif tag == 'B-Location':
                # Save any previous entity first
                if current_entity_tokens and current_entity_type:
                    entity_text = ' '.join(current_entity_tokens)
                    entities.append({
                        'dipl_text': entity_text,
                        'label': current_entity_type
                    })
                # Start new Location entity
                current_entity_tokens = [token]
                current_entity_type = 'LOC'
                
            # Check for I-Person or I-Location (Inside/continuation of entity)
            elif tag == 'I-Person' and current_entity_type == 'PER':
                current_entity_tokens.append(token)
                
            elif tag == 'I-Location' and current_entity_type == 'LOC':
                current_entity_tokens.append(token)
                
            # Any other tag (O, or different entity type) ends current entity
            else:
                if current_entity_tokens and current_entity_type:
                    entity_text = ' '.join(current_entity_tokens)
                    entities.append({
                        'dipl_text': entity_text,
                        'label': current_entity_type
                    })
                current_entity_tokens = []
                current_entity_type = None
    
    # Don't forget any entity at end of file
    if current_entity_tokens and current_entity_type:
        entity_text = ' '.join(current_entity_tokens)
        entities.append({
            'dipl_text': entity_text,
            'label': current_entity_type
        })
    
    return entities


def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Remove duplicate entities while preserving order."""
    seen = set()
    unique_entities = []
    for entity in entities:
        key = (entity['dipl_text'], entity['label'])
        if key not in seen:
            seen.add(key)
            unique_entities.append(entity)
    return unique_entities


def process_directory(input_dir: str, output_dir: str, deduplicate: bool = False):
    """
    Process all .txt files in a directory and output separate JSON files.
    
    Args:
        input_dir: Directory containing BIO format .txt files
        output_dir: Directory for output JSON files
        deduplicate: If True, remove duplicate entities within each file
    """
    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files
    txt_files = sorted(Path(input_dir).glob('*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} files to process...")
    print("-" * 70)
    
    grand_total_per = 0
    grand_total_loc = 0
    grand_total_raw = 0
    grand_total_dedup = 0
    
    for filepath in txt_files:
        entities = parse_bio_file(str(filepath))
        raw_count = len(entities)
        grand_total_raw += raw_count
        
        if deduplicate:
            entities = deduplicate_entities(entities)
        
        # Count stats
        per_count = sum(1 for e in entities if e['label'] == 'PER')
        loc_count = sum(1 for e in entities if e['label'] == 'LOC')
        grand_total_per += per_count
        grand_total_loc += loc_count
        grand_total_dedup += len(entities)
        
        # Output JSON file (same name but .json extension)
        output_filename = filepath.stem + '.json'
        output_path = Path(output_dir) / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        
        if deduplicate:
            print(f"  {filepath.name} -> {output_filename}: {per_count} PER, {loc_count} LOC (dedup: {raw_count} -> {len(entities)})")
        else:
            print(f"  {filepath.name} -> {output_filename}: {per_count} PER, {loc_count} LOC")
    
    print("-" * 70)
    print(f"\nTotal extracted across all files:")
    if deduplicate:
        print(f"  Raw entities: {grand_total_raw}")
        print(f"  After deduplication: {grand_total_dedup}")
    print(f"  PERSON entities: {grand_total_per}")
    print(f"  LOCATION entities: {grand_total_loc}")
    print(f"\nSaved {len(txt_files)} JSON files to: {output_dir}")


if __name__ == '__main__':
    process_directory(INPUT_DIR, OUTPUT_DIR, DEDUPLICATE)