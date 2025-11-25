#!/usr/bin/env python3
"""
Extract NPR entities from PSD format with title detection.
Handles both adjacent NPR tags and NP-PRN appositive structures (including multi-line).
"""

import json
import re


# Title words that indicate roles
TITLES = {
    'konungur', 'konungi', 'konungs', 'konung',  # king
    'hersir',  # chieftain
    'drottning', 'drottningar',  # queen
    'jarl', 'jarli', 'jarls',  # earl
    'biskup', 'biskupi', 'biskups',  # bishop
}


def is_title(lemma):
    """Check if a lemma is a title/role."""
    return lemma.lower() in TITLES


def extract_entities(input_file):
    """
    Extract named entities from PSD file.
    Handles both:
    1. Adjacent NPR tags: (NPR-D Aðalsteini-aðalsteinn) (NPR-D konungi-konungur)
    2. NP-PRN appositives: (NPR-N Eiríkur-eiríkur) (NP-PRN (N-N konungur-konungur))
    """
    entities = []
    seen = set()
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove newlines to handle multi-line structures, but keep sentence boundaries
    # Split by sentence IDs to process each sentence as one unit
    sentences = re.split(r'\(ID [^\)]+\)', content)
    
    for sentence in sentences:
        # Collapse whitespace to single line for easier pattern matching
        sentence = ' '.join(sentence.split())
        
        if not sentence.strip():
            continue
        
        # Pattern 1: Adjacent NPR tags (original handling)
        adjacent_pattern = r'\(NPR-[NDGA]\s+(\S+?)-(\S+?)\)'
        adjacent_matches = list(re.finditer(adjacent_pattern, sentence))
        
        # Track positions we've processed
        processed_positions = set()
        
        # Check for adjacent NPR pairs
        for i, match in enumerate(adjacent_matches):
            if i in processed_positions:
                continue
                
            word, lemma = match.groups()
            
            # Check if followed by another NPR that's a title
            if i + 1 < len(adjacent_matches):
                next_match = adjacent_matches[i + 1]
                # Check they're actually adjacent (within reasonable distance)
                distance = next_match.start() - match.end()
                if distance < 50:  # Allow some whitespace/parens
                    next_word, next_lemma = next_match.groups()
                    
                    # name + title
                    if is_title(next_lemma):
                        if word.lower() not in seen:
                            entities.append({"dipl_text": word, "label": "PER"})
                            seen.add(word.lower())
                        
                        combined = f"{word} {next_word}"
                        if combined.lower() not in seen:
                            entities.append({"dipl_text": combined, "label": "PER-ROL"})
                            seen.add(combined.lower())
                        
                        processed_positions.add(i)
                        processed_positions.add(i + 1)
                        continue
                    
                    # title + name
                    elif is_title(lemma):
                        if next_word.lower() not in seen:
                            entities.append({"dipl_text": next_word, "label": "PER"})
                            seen.add(next_word.lower())
                        
                        combined = f"{word} {next_word}"
                        if combined.lower() not in seen:
                            entities.append({"dipl_text": combined, "label": "PER-ROL"})
                            seen.add(combined.lower())
                        
                        processed_positions.add(i)
                        processed_positions.add(i + 1)
                        continue
            
            # Not a standalone title
            if not is_title(lemma) and i not in processed_positions:
                if word.lower() not in seen:
                    entities.append({"dipl_text": word, "label": ""})
                    seen.add(word.lower())
        
        # Pattern 2: NP-PRN appositive structure (now handles multi-line)
        # (NPR-CASE name-lemma) ... (NP-PRN (N-CASE title-lemma))
        # Allow for whitespace and nested parens between NPR and NP-PRN
        appositive_pattern = r'\(NPR-[NDGA]\s+(\S+?)-(\S+?)\)[^\(]*\(NP-PRN\s+\(N-[NDGA]\s+(\S+?)-(\S+?)\)'
        
        for match in re.finditer(appositive_pattern, sentence):
            name_word, name_lemma, title_word, title_lemma = match.groups()
            
            # Only process if the second word is a title
            if is_title(title_lemma):
                # Add name as PER
                if name_word.lower() not in seen:
                    entities.append({"dipl_text": name_word, "label": "PER"})
                    seen.add(name_word.lower())
                
                # Add name + title as PER-ROL
                combined = f"{name_word} {title_word}"
                if combined.lower() not in seen:
                    entities.append({"dipl_text": combined, "label": "PER-ROL"})
                    seen.add(combined.lower())
    
    return entities


def main():
    input_file = "ihpc/1275_morkin/1275.morkin.nar-his.psd.txt"
    output_file = "ihpc/1275_morkin/1275_morkin_entities_init.json"
    
    print("Extracting entities from PSD format...")
    print("Handling adjacent NPR and multi-line NP-PRN structures...")
    entities = extract_entities(input_file)
    
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    
    # Statistics
    labeled_count = sum(1 for e in entities if e['label'])
    unlabeled_count = sum(1 for e in entities if not e['label'])
    per_count = sum(1 for e in entities if e['label'] == 'PER')
    per_rol_count = sum(1 for e in entities if e['label'] == 'PER-ROL')
    
    print(f"\n✓ Extracted {len(entities)} unique entities")
    print(f"  - {per_count} labeled as PER")
    print(f"  - {per_rol_count} labeled as PER-ROL")
    print(f"  - {unlabeled_count} with empty label (for manual classification)")
    print(f"\n✓ Saved to: {output_file}")
    
    # Show samples
    print("\n--- Sample PER entities ---")
    for e in [e for e in entities if e['label'] == 'PER'][:10]:
        print(f"  {e['dipl_text']}")
    
    print("\n--- Sample PER-ROL entities ---")
    for e in [e for e in entities if e['label'] == 'PER-ROL'][:10]:
        print(f"  {e['dipl_text']}")
    
    print("\n--- Sample unlabeled entities ---")
    for e in [e for e in entities if not e['label']][:10]:
        print(f"  {e['dipl_text']}")


if __name__ == "__main__":
    main()