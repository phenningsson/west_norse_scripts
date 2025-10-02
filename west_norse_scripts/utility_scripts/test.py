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
    Finds positions of entities in clean text based solely on diplomatic text.
    Returns the found positions and statistics.
    """
    results = []
    position_map = {}  # To track which positions we've already assigned
    not_found = 0
    total_entities = len(entities)
    clean_text_normalized = normalize_dipl_text(clean_text)

    # First collect all unique entities by their text
    unique_entities = {}
    for entity in entities:
        text = entity['dipl_text']
        if text not in unique_entities:
            unique_entities[text] = entity
        # If we have duplicates, we'll just keep the first one

    # Now process each unique entity
    for i, (text, entity) in enumerate(unique_entities.items(), 1):
        search_text = entity['dipl_text']
        if not search_text:
            print(f"Warning: Empty dipl_text for entity {i}/{len(unique_entities)}")
            not_found += 1
            continue

        search_text_norm = normalize_dipl_text(search_text)
        search_text_len = len(search_text)

        # Try exact matching first (case-sensitive)
        pos = 0
        while True:
            pos = clean_text.find(search_text, pos)
            if pos == -1:
                break

            # Verify that the matched text exactly matches our search text
            matched_text = clean_text[pos:pos+len(search_text)]
            if matched_text == search_text:
                # Check if we've already assigned an entity at this position
                if pos not in position_map:
                    results.append({
                        'text': matched_text,
                        'start': pos,
                        'end': pos + len(search_text) - 1,
                        'label': entity['label'],
                        'subtype': entity.get('subtype'),
                        'source': 'exact_match'
                    })
                    position_map[pos] = True  # Mark this position as taken
                else:
                    print(f"Duplicate entity '{matched_text}' at position {pos}")
                pos += 1  # Move past this match to find others
            else:
                pos += 1

        # If not found exactly, try normalized matching (more permissive)
        if not any(pos in position_map for pos in range(len(clean_text))):
            norm_pos = 0
            while True:
                norm_pos = clean_text_normalized.find(search_text_norm, norm_pos)
                if norm_pos == -1:
                    break

                window_size = max(100, len(search_text) * 2)
                start = max(0, norm_pos - window_size//2)
                end = min(len(clean_text), norm_pos + len(search_text_norm) + window_size//2)
                context = clean_text[start:end]

                context_pos = context.find(search_text)
                if context_pos != -1:
                    actual_pos = start + context_pos
                    if actual_pos not in position_map:
                        matched_text = clean_text[actual_pos:actual_pos+len(search_text)]
                        # Verify this match wasn't already found by exact match
                        if not any(r['start'] == actual_pos and r['end'] == actual_pos + len(search_text) - 1 for r in results):
                            results.append({
                                'text': matched_text,
                                'start': actual_pos,
                                'end': actual_pos + len(search_text) - 1,
                                'label': entity['label'],
                                'subtype': entity.get('subtype'),
                                'source': 'normalized_match'
                            })
                            position_map[actual_pos] = True  # Mark position as taken
                    norm_pos += len(search_text_norm)
                    continue
                norm_pos += 1

        # Try partial matching for multi-word entities
        if not any(pos in position_map for pos in range(len(clean_text))):
            if len(search_text.split()) > 1:
                first_word = search_text.split()[0]
                first_pos = clean_text.find(first_word)
                if first_pos != -1:
                    remaining_text = clean_text[first_pos + len(first_word):first_pos + len(search_text) + 50]
                    remaining_search = search_text[len(first_word):].strip()
                    remaining_pos = remaining_text.find(remaining_search)
                    if remaining_pos != -1:
                        full_pos = first_pos
                        full_length = len(search_text)
                        matched_text = clean_text[full_pos:full_pos+full_length]
                        if normalize_dipl_text(matched_text) == search_text_norm and full_pos not in position_map:
                            results.append({
                                'text': matched_text,
                                'start': full_pos,
                                'end': full_pos + full_length - 1,
                                'label': entity['label'],
                                'subtype': entity.get('subtype'),
                                'source': 'partial_match'
                            })
                            position_map[full_pos] = True  # Mark position as taken

    # Print summary
    found_count = len(results)
    print(f"\nResults: Found {found_count}/{len(unique_entities)} unique entity occurrences")
    if not_found > 0:
        print(f"Failed to find {not_found} entities ({not_found/len(unique_entities):.1%})")

    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def main():
    # File paths
    clean_text_file = 'AM_162_B_κ_fol_cleaned.txt'
    entities_file = 'dipl_entities.json'
    output_file = 'am162_entities_with_positions.json'

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
    entity_counts = defaultdict(int)
    for entity in entities:
        key = (entity['dipl_text'], entity['label'], entity.get('subtype', ''))
        entity_counts[key] += 1

    duplicates = [key for key, count in entity_counts.items() if count > 1]
    if duplicates:
        print("\nDuplicate entities found in input file:")
        for key in duplicates:
            print(f"'{key[0]}' ({key[1]}-{key[2] if key[2] else 'none'}): {entity_counts[key]} occurrences")
    else:
        print("\nNo duplicate entities found in input file")

    # Find entity positions
    print("\nFinding entity positions (diplomatic text only)...")
    entity_positions = find_entity_positions(clean_text, entities, output_file)

    # Print statistics
    if entity_positions:
        labels = {}
        subtypes = {}
        sources = {}

        for entity in entity_positions:
            label = entity['label']
            subtype = entity.get('subtype', 'none')
            source = entity.get('source', 'unknown')
            labels[label] = labels.get(label, 0) + 1
            key = (label, subtype)
            subtypes[key] = subtypes.get(key, 0) + 1
            sources[source] = sources.get(source, 0) + 1

        print("\n=== STATISTICS ===")
        print(f"Total unique entity positions found: {len(entity_positions)}")

        print("\nBy label:")
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")

        print("\nBy subtype:")
        # Sort subtypes properly
        subtype_items = sorted(subtypes.items(), key=lambda x: (x[0][0], x[0][1]))
        for (label, subtype), count in subtype_items:
            if subtype != 'none':
                print(f"  {label}-{subtype}: {count}")

        print("\nBy source:")
        for source, count in sorted(sources.items()):
            print(f"  {source}: {count}")

        # Print first 10 found entities for verification
        print("\nFirst 10 unique entity positions found:")
        for i, entity in enumerate(entity_positions[:10], 1):
            subtype_display = f"-{entity.get('subtype', '')}" if entity.get('subtype') else ''
            print(f"{i}. '{entity['text']}' ({entity['label']}{subtype_display}) at {entity['start']}-{entity['end']}")

if __name__ == "__main__":
    main()
