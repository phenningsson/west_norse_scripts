import json
import re
from collections import defaultdict
from unicodedata import normalize, category

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

def count_entity_occurrences(clean_text, entities):
    """
    Counts how many times each entity appears in the clean text.
    Returns a dictionary mapping entity texts to their occurrence counts.
    """
    counts = defaultdict(int)
    multiple_matches = []

    for entity in entities:
        search_text = entity['dipl_text']
        if not search_text:
            continue

        # Count exact matches
        count = 0
        pos = 0
        while True:
            pos = clean_text.find(search_text, pos)
            if pos == -1:
                break
            matched_text = clean_text[pos:pos+len(search_text)]
            if matched_text == search_text:
                count += 1
                pos += 1  # Move past this match
            else:
                pos += 1  # Continue searching

        counts[search_text] = count
        if count > 1:
            multiple_matches.append({
                'text': search_text,
                'label': entity['label'],
                'subtype': entity.get('subtype', ''),
                'count': count
            })

    return counts, multiple_matches

def find_entity_positions(clean_text, entities, output_file):
    """
    Finds positions of entities in clean text based solely on diplomatic text.
    Returns the found positions and statistics.
    """
    results = []
    not_found = 0
    total_entities = len(entities)
    clean_text_normalized = normalize_dipl_text(clean_text)

    # First count occurrences of each entity
    occurrence_counts, multiple_matches = count_entity_occurrences(clean_text, entities)

    print("\nEntities that match multiple times:")
    for match in sorted(multiple_matches, key=lambda x: x['count'], reverse=True):
        subtype_display = f"-{match['subtype']}" if match['subtype'] else ""
        print(f"'{match['text']}' ({match['label']}{subtype_display}): {match['count']} occurrences")

    for i, entity in enumerate(entities, 1):
        search_text = entity['dipl_text']
        if not search_text:
            print(f"Warning: Empty dipl_text for entity {i}/{total_entities}")
            not_found += 1
            continue

        search_text_norm = normalize_dipl_text(search_text)
        search_text_len = len(search_text)
        found = False

        # First try exact match
        pos = 0
        while True:
            pos = clean_text.find(search_text, pos)
            if pos == -1:
                break

            matched_text = clean_text[pos:pos+len(search_text)]
            if matched_text == search_text:
                results.append({
                    'text': matched_text,
                    'start': pos,
                    'end': pos + len(search_text) - 1,
                    'label': entity['label'],
                    'subtype': entity.get('subtype'),
                    'source': 'exact_match'
                })
                found = True
                pos += 1  # Move past this match to find others
            else:
                pos += 1

        # Rest of your matching logic (normalized and partial matches)...
        if not found and search_text_norm:
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
                    matched_text = clean_text[actual_pos:actual_pos+len(search_text)]
                    if not any(r['start'] == actual_pos and r['end'] == actual_pos + len(search_text) - 1 for r in results):
                        results.append({
                            'text': matched_text,
                            'start': actual_pos,
                            'end': actual_pos + len(search_text) - 1,
                            'label': entity['label'],
                            'subtype': entity.get('subtype'),
                            'source': 'normalized_match'
                        })
                        found = True
                        norm_pos += len(search_text_norm)
                        continue
                norm_pos += 1

        if not found and len(search_text.split()) > 1:
            # Partial matching logic remains the same
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
                    if normalize_dipl_text(matched_text) == search_text_norm:
                        results.append({
                            'text': matched_text,
                            'start': full_pos,
                            'end': full_pos + full_length - 1,
                            'label': entity['label'],
                            'subtype': entity.get('subtype'),
                            'source': 'partial_match'
                        })
                        found = True

        if not found:
            print(f"\nWarning: Could not find entity '{search_text}' (normalized: '{search_text_norm}')")
            not_found += 1

    # Print summary statistics
    found_count = len(results)
    print(f"\nResults: Found {found_count}/{total_entities} entities")
    if not_found > 0:
        print(f"Failed to find {not_found} entities ({not_found/total_entities:.1%})")

    # Analyze PAT entities specifically
    original_pat_count = sum(1 for e in entities if e.get('subtype') == 'PAT')
    positioned_pat_count = sum(1 for e in results if e.get('subtype') == 'PAT')
    print(f"\nPAT entity count comparison:")
    print(f"Original file: {original_pat_count} PAT entities")
    print(f"Positioned file: {positioned_pat_count} PAT entities")
    print(f"Difference: {positioned_pat_count - original_pat_count}")

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

    # Count original PAT entities
    original_pat_count = sum(1 for e in entities if e.get('subtype') == 'PAT')
    print(f"Original PAT entity count: {original_pat_count}")

    # Find entity positions
    print("\nFinding entity positions (diplomatic text only)...")
    entity_positions = find_entity_positions(clean_text, entities, output_file)

    # Print statistics if we have results
    if entity_positions:
        labels = {}
        subtypes = {}
        sources = {}

        # Initialize counts
        for entity in entity_positions:
            label = entity['label']
            subtype = entity.get('subtype', '')  # Default to empty string if no subtype
            source = entity.get('source', 'unknown')

            labels[label] = labels.get(label, 0) + 1
            key = (label, subtype)  # Use tuple as key
            subtypes[key] = subtypes.get(key, 0) + 1
            sources[source] = sources.get(source, 0) + 1

        print("\n=== STATISTICS ===")
        print(f"Total entities found: {len(entity_positions)}")

        # Print by label
        print("\nBy label:")
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")

        # Print by subtype - handle None cases by sorting properly
        print("\nBy subtype:")
        # Convert to list of tuples for sorting
        subtype_list = []
        for (label, subtype), count in subtypes.items():
            if subtype:  # Only print if there's a subtype
                subtype_list.append(((label, subtype), count))

        # Sort alphabetically by label then subtype
        subtype_list.sort(key=lambda x: (x[0][0], x[0][1]))

        for (label, subtype), count in subtype_list:
            print(f"  {label}-{subtype}: {count}")

        # Print by source
        print("\nBy source:")
        for source, count in sorted(sources.items()):
            print(f"  {source}: {count}")

        # Print first 10 found entities for verification
        print("\nFirst 10 entities found:")
        for i, entity in enumerate(entity_positions[:10], 1):
            subtype_display = ''
            if 'subtype' in entity and entity['subtype']:
                subtype_display = f"-{entity['subtype']}"
            print(f"{i}. '{entity['text']}' ({entity['label']}{subtype_display}) at {entity['start']}-{entity['end']}")

if __name__ == "__main__":
    main()