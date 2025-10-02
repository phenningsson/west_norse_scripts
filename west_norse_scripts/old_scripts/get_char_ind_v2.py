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
    Handles entities with trailing punctuation by ignoring punctuation after the entity.
    Allows multiple entities at same position if they have different types/subtypes.
    Prevents duplicate entries for same entity type at same position.
    """
    results = []
    position_entity_map = defaultdict(list)
    not_found = 0
    total_entities = len(entities)
    clean_text_normalized = normalize_dipl_text(clean_text)

    # Define punctuation that might appear after entities
    trailing_punctuation = r'[.,;:!?)\]]'

    # First create a mapping of unique entity texts to all their variants
    entity_variants = defaultdict(list)
    for entity in entities:
        text = entity['dipl_text']
        if text:  # Skip empty texts
            entity_variants[text].append(entity)

    processed_positions = set()  # To track which positions we've already processed

    # Now process each unique entity text
    for search_text, variants in entity_variants.items():
        search_text_norm = normalize_dipl_text(search_text)
        search_text_len = len(search_text)

        # Try exact matching first (case-sensitive), allowing for trailing punctuation
        pos = 0
        while pos < len(clean_text):
            # First try exact match
            match_pos = clean_text.find(search_text, pos)
            if match_pos == -1:
                break

            # Check if this is followed by punctuation
            end_pos = match_pos + search_text_len
            if end_pos < len(clean_text) and re.match(trailing_punctuation, clean_text[end_pos]):
                # We found the entity followed by punctuation - use this position
                matched_text = search_text  # Without the punctuation
                punctuation = clean_text[end_pos]
            else:
                # Either no punctuation or punctuation is part of the entity
                matched_text = clean_text[match_pos:match_pos+search_text_len]
                punctuation = ''

            # Only consider it a match if we found the exact entity text
            if matched_text == search_text:
                # For each variant of this entity text, add it to results
                for variant in variants:
                    entity_key = (variant['label'], variant.get('subtype', ''), match_pos, match_pos + search_text_len - 1)

                    # Check if we've already added this exact entity at this position
                    if entity_key not in processed_positions:
                        results.append({
                            'text': matched_text,
                            'start': match_pos,
                            'end': match_pos + search_text_len - 1,
                            'label': variant['label'],
                            'subtype': variant.get('subtype'),
                            'source': 'exact_match',
                            'trailing_punctuation': punctuation
                        })
                        processed_positions.add(entity_key)

                # Move past this match (and the punctuation if any)
                pos = match_pos + search_text_len + (1 if punctuation else 0)
            else:
                pos += 1

        # If not found exactly, try normalized matching (more permissive)
        norm_pos = 0
        while True:
            norm_pos = clean_text_normalized.find(search_text_norm, norm_pos)
            if norm_pos == -1:
                break

            window_size = max(100, len(search_text) * 2)
            start = max(0, norm_pos - window_size//2)
            end = min(len(clean_text), norm_pos + len(search_text_norm) + window_size//2)
            context = clean_text[start:end]

            # Find the exact entity text in this context
            context_pos = context.find(search_text)
            if context_pos != -1:
                actual_pos = start + context_pos
                matched_text = context[context_pos:context_pos+search_text_len]

                # Check for trailing punctuation
                context_end = actual_pos + search_text_len
                if context_end < len(clean_text) and re.match(trailing_punctuation, clean_text[context_end]):
                    punctuation = clean_text[context_end]
                    entity_end = context_end - 1
                else:
                    punctuation = ''
                    entity_end = context_end - 1

                # For each variant of this entity text, add it to results
                for variant in variants:
                    entity_key = (variant['label'], variant.get('subtype', ''), actual_pos, entity_end)

                    if entity_key not in processed_positions:
                        results.append({
                            'text': search_text,  # Always use the original entity text
                            'start': actual_pos,
                            'end': entity_end,
                            'label': variant['label'],
                            'subtype': variant.get('subtype'),
                            'source': 'normalized_match',
                            'trailing_punctuation': punctuation
                        })
                        processed_positions.add(entity_key)

                norm_pos = context_end + (1 if punctuation else 0)
                continue

            norm_pos += 1

        # Try partial matching for multi-word entities
        if len(search_text.split()) > 1:
            first_word = search_text.split()[0]
            first_pos = clean_text.find(first_word)
            if first_pos != -1:
                remaining_text = clean_text[first_pos + len(first_word):first_pos + len(search_text) + 50]
                remaining_search = search_text[len(first_word):].strip()
                remaining_pos = remaining_text.find(remaining_search)

                if remaining_pos != -1:
                    full_pos = first_pos
                    matched_text = clean_text[full_pos:full_pos+len(search_text)]

                    # Check for trailing punctuation
                    full_end = full_pos + len(search_text)
                    if full_end < len(clean_text) and re.match(trailing_punctuation, clean_text[full_end]):
                        punctuation = clean_text[full_end]
                        entity_end = full_end - 1
                    else:
                        punctuation = ''
                        entity_end = full_end - 1

                    if normalize_dipl_text(clean_text[full_pos:entity_end+1]) == search_text_norm:
                        # For each variant of this entity text, add it to results
                        for variant in variants:
                            entity_key = (variant['label'], variant.get('subtype', ''), full_pos, entity_end)

                            if entity_key not in processed_positions:
                                results.append({
                                    'text': search_text,  # Use original entity text
                                    'start': full_pos,
                                    'end': entity_end,
                                    'label': variant['label'],
                                    'subtype': variant.get('subtype'),
                                    'source': 'partial_match',
                                    'trailing_punctuation': punctuation
                                })
                                processed_positions.add(entity_key)

    # Calculate statistics
    found_count = len(results)
    original_count = len(entities)
    print(f"\nResults: Found {found_count} entity occurrences from {original_count} unique entities")
    print(f"Failed to find {not_found} entities")

    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results, found_count, original_count

def main():
    # File paths
    clean_text_file = 'olafs_saga_helga/olafs_saga_helga_clean.txt'
    entities_file = 'dipl_olaf_entities.json'
    output_file = 'olaf_entities_with_positions.json'

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
        sources = defaultdict(int)

        for entity in entity_positions:
            label = entity['label']
            subtype = entity.get('subtype', 'none')
            source = entity.get('source', 'unknown')
            labels[label] += 1
            subtypes[(label, subtype)] += 1
            sources[source] += 1

        print("\n=== STATISTICS ===")
        print(f"Total entity occurrences found: {found_count}")
        print(f"From {original_count} unique entity definitions")

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
            punct_info = f" + '{entity.get('trailing_punctuation', '')}'" if entity.get('trailing_punctuation') else ''
            print(f"{i}. '{entity['text']}' ({entity['label']}{subtype_display}) at {entity['start']}-{entity['end']}{punct_info}")

if __name__ == "__main__":
    main()
