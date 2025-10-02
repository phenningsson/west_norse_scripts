import json
import re
from collections import defaultdict

def normalize(text):
    """Basic normalization for comparison."""
    # Remove XML tags if any remain
    text = re.sub(r'<[^>]+>', '', text)
    # Normalize whitespace but preserve line breaks
    text = re.sub(r'[^\S\n]+', ' ', text).strip()
    return text.lower()  # Case-insensitive comparison

def find_mismatches(clean_text_file, entities_file, output_file):
    """
    Finds mismatches between entities in the JSON file and the clean text.
    Saves results to output_file.
    """
    # Load clean text
    with open(clean_text_file, 'r', encoding='utf-8') as f:
        clean_text = f.read()

    # Load simplified entities
    with open(entities_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    clean_text_norm = normalize(clean_text)
    results = []

    for i, entity in enumerate(entities, 1):
        dipl_text = entity['dipl_text']
        norm_dipl = normalize(dipl_text)

        # Check if normalized entity appears in normalized text
        found_in_norm = norm_dipl in clean_text_norm

        # Check for exact match in original text
        exact_pos = clean_text.find(dipl_text)
        exact_match = exact_pos != -1

        # Try to find a close match if not exact
        close_match = False
        if found_in_norm and not exact_match:
            # Try to find where in the text the normalized version appears
            norm_start = clean_text_norm.find(norm_dipl)
            if norm_start != -1:
                # Get context around the normalized match
                context_start = max(0, norm_start - 50)
                context_end = min(len(clean_text_norm), norm_start + len(norm_dipl) + 50)
                context = clean_text[context_start:context_end]

                # Try to find the closest match in this context
                # This is a simple approach - could be enhanced with fuzzy matching
                for window_size in [5, 10, 20]:  # Try different window sizes
                    for offset in range(-window_size, window_size+1):
                        start = max(0, norm_start + offset)
                        end = start + len(dipl_text)
                        if end <= len(clean_text):
                            candidate = clean_text[start:end]
                            if normalize(candidate) == norm_dipl:
                                close_match = True
                                break
                    if close_match:
                        break

        results.append({
            'entity_index': i,
            'dipl_text': dipl_text,
            'label': entity['label'],
            'subtype': entity.get('subtype', ''),
            'exists_in_norm_text': found_in_norm,
            'exact_match_in_original': exact_match,
            'has_close_match': close_match if not exact_match else False,
            'norm_dipl_text': norm_dipl
        })

        if not found_in_norm:
            print(f"Not found: '{dipl_text}'")
        elif not exact_match:
            print(f"Not exact: '{dipl_text}'")

    # Save mismatch report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary statistics
    not_found = sum(1 for r in results if not r['exists_in_norm_text'])
    no_exact_match = sum(1 for r in results if r['exists_in_norm_text'] and not r['exact_match_in_original'])
    close_matches = sum(1 for r in results if not r['exact_match_in_original'] and r.get('has_close_match', False))

    print("\n=== MISMATCH ANALYSIS RESULTS ===")
    print(f"Total entities analyzed: {len(results)}")
    print(f"Not found in normalized text: {not_found}")
    print(f"Found in normalized but not exact match: {no_exact_match}")
    print(f"Close matches found: {close_matches}") 

if __name__ == "__main__":
    clean_text_file = 'AM_162_B_κ_fol_cleaned.txt'
    entities_file = 'dipl_entities.json'
    output_file = 'entity_mismatch_report.json'
    find_mismatches(clean_text_file, entities_file, output_file)
