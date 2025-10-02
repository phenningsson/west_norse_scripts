import json
import re
from unicodedata import normalize, category
from collections import defaultdict

def debug_print(text, max_length=100):
    """Helper function to print text with visible representation of whitespace and special chars"""
    debug_text = []
    for c in text:
        if c == ' ':
            debug_text.append('␣')  # visible space
        elif c == '\t':
            debug_text.append('⇥')
        elif c == '\n':
            debug_text.append('⏎')
        elif category(c).startswith('C'):  # control character
            debug_text.append(f'\\u{ord(c):04x}')
        else:
            debug_text.append(c)
    debug_str = ''.join(debug_text)
    if len(debug_str) > max_length:
        debug_str = debug_str[:max_length] + '...'
    print(f"Debug: '{debug_str}' (length: {len(text)})")
    return debug_str

def normalize_dipl_text(text):
    """Normalize diplomatic text for comparison, keeping only essential normalization"""
    text = normalize('NFKC', text)
    text = re.sub(r'[^\S\n]+', ' ', text)  # Preserve newlines
    return text.strip()

def find_entity_positions(clean_text, entities, output_file):
    """
    Finds positions of entities in clean text based solely on diplomatic text,
    ensuring we only match complete words (not substrings of longer words).
    """
    results = []
    seen_positions = set()  # To track which (start, end, label, subtype) combinations we've already processed

    # Create a mapping from entity text to all its variants
    entity_variants = defaultdict(list)
    for entity in entities:
        text = entity['dipl_text']
        if text:
            entity_variants[text].append(entity)

    # Process each unique entity text
    for search_text in entity_variants:
        variants = entity_variants[search_text]
        search_len = len(search_text)

        # Try to find all occurrences of this entity text
        pos = 0
        while pos < len(clean_text):
            pos = clean_text.find(search_text, pos)
            if pos == -1:
                break

            # Check if this is a complete word match (followed by word boundary or punctuation)
            end_pos = pos + search_len

            # Check if we're at a word boundary
            is_word_boundary = False

            # Check if we're at the end of the text
            if end_pos == len(clean_text):
                is_word_boundary = True
            else:
                next_char = clean_text[end_pos]
                # Check for word boundaries (space or punctuation)
                if next_char in ' ,;:!?).]}' or category(next_char).startswith('P'):
                    is_word_boundary = True

            if is_word_boundary:
                # For each variant of this entity text, add it to results if not already seen
                for variant in variants:
                    entity_key = (pos, end_pos - 1, variant['label'], variant.get('subtype', 'none'))
                    if entity_key not in seen_positions:
                        results.append({
                            'text': search_text,
                            'start': pos,
                            'end': end_pos - 1,
                            'label': variant['label'],
                            'subtype': variant.get('subtype'),
                            'source': 'exact_match'
                        })
                        seen_positions.add(entity_key)

            pos += 1  # Move past this position to find others

    # Calculate statistics
    found_count = len(results)
    original_count = len(entities)
    print(f"\nResults: Found {found_count} entity occurrences from {original_count} unique entities")

    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results, found_count, original_count

def main():
    # File paths
    clean_text_file = 'drauma_jons_clean.txt'
    entities_file = 'drauma_jons_dipl_entities_dipl.json'
    output_file = 'drauma_jons_entities_with_positions.json'

    # Load clean text
    print(f"Loading clean text from {clean_text_file}...")
    with open(clean_text_file, 'r', encoding='utf-8') as file:
        clean_text = file.read()

    # Print some info about the clean text
    print("\n=== CLEAN TEXT INFO ===")
    print(f"Total length: {len(clean_text)} characters")
    debug_print(clean_text[:200])

    # Load entities
    print(f"\nLoading entities from {entities_file}...")
    with open(entities_file, 'r', encoding='utf-8') as file:
        entities = json.load(file)
    print(f"Loaded {len(entities)} entities")

    # Check for duplicate entities in input
    entity_text_count = defaultdict(int)
    duplicate_entities = defaultdict(list)
    for i, entity in enumerate(entities):
        key = (entity['dipl_text'], entity['label'], entity.get('subtype', ''))
        entity_text_count[key] += 1
        duplicate_entities[key].append(i)

    duplicates = [(key, count) for key, count in entity_text_count.items() if count > 1]
    if duplicates:
        print("\nDuplicate entities found in input file:")
        for (text, label, subtype), count in duplicates:
            print(f"'{text}' ({label}-{subtype if subtype else 'none'}): {count} occurrences (indices: {duplicate_entities[(text, label, subtype)]})")
    else:
        print("\nNo duplicate entities found in input file")

    # Find entity positions
    print("\nFinding entity positions (diplomatic text only)...")
    entity_positions, found_count, original_count = find_entity_positions(clean_text, entities, output_file)

    # Print statistics
    if entity_positions:
        labels = defaultdict(int)
        subtypes = defaultdict(int)
        for entity in entity_positions:
            label = entity['label']
            subtype = entity.get('subtype', 'none')
            labels[label] += 1
            subtypes[(label, subtype)] += 1

        print("\n=== STATISTICS ===")
        print(f"Total entity occurrences found: {found_count}")
        print(f"From {original_count} unique entity definitions")

        print("\nBy label:")
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")

        print("\nBy subtype:")
        # Sort subtypes properly, handling None values
        subtype_items = []
        for (label, subtype), count in subtypes.items():
            # Convert None to empty string for consistent sorting
            subtype_str = subtype if subtype is not None else ''
            subtype_items.append(((label, subtype_str), count))

        # Sort by label then subtype
        subtype_items.sort(key=lambda x: (x[0][0], x[0][1]))
        for (label, subtype), count in subtype_items:
            # Display 'none' instead of empty string for None subtypes
            display_subtype = subtype if subtype != '' else 'none'
            print(f"  {label}-{display_subtype}: {count}")

        # Find positions with multiple entities
        position_counts = defaultdict(int)
        for entity in entity_positions:
            position_counts[(entity['start'], entity['end'])] += 1

        overlapping_positions = [(pos, count) for pos, count in position_counts.items() if count > 1]
        if overlapping_positions:
            print("\nPositions with multiple entities (overlapping annotations):")
            for (start, end), count in overlapping_positions:
                print(f"  Positions {start}-{end}: {count} entities")
                # Show the entities at this position
                for entity in entity_positions:
                    if entity['start'] == start and entity['end'] == end:
                        subtype_display = f"-{entity.get('subtype', '')}" if entity.get('subtype') else ''
                        print(f"    '{entity['text']}' ({entity['label']}{subtype_display})")
        else:
            print("\nNo positions with multiple entities found")

        # Print first 10 found entities for verification
        print("\nFirst 10 entity occurrences found:")
        for i, entity in enumerate(entity_positions[:10], 1):
            subtype_display = f"-{entity.get('subtype', '')}" if entity.get('subtype') else ''
            print(f"{i}. '{entity['text']}' ({entity['label']}{subtype_display}) at {entity['start']}-{entity['end']}")

if __name__ == "__main__":
    main()
