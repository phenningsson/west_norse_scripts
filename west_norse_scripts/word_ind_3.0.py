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

def get_word_positions(clean_text, output_file):
    """
    Finds positions of all words in clean text.
    Returns list of dictionaries with word information and character positions.
    """
    results = []
    word_pattern = re.compile(r'\w+[\w\'-]*\w+|\w')  # Match words with optional apostrophes/hyphens

    # Find all word matches in the text
    for match in word_pattern.finditer(clean_text):
        word = match.group()
        start_pos = match.start()
        end_pos = match.end() - 1  # Convert to inclusive end position

        results.append({
            'text': word,
            'start': start_pos,
            'end': end_pos,
        })

    # Calculate statistics
    print(f"\nResults: Found {len(results)} words in the text")

    # Save results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results

def main():
    # File paths
    clean_text_file = '/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/olafs_saga_helga/olafs_saga_normalised.txt'
    output_file = '/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/olafs_saga_helga/olafs_saga_normalised_word_positions.json'

    # Load clean text
    print(f"Loading clean text from {clean_text_file}...")
    with open(clean_text_file, 'r', encoding='utf-8') as file:
        clean_text = file.read()

    # Print some info about the clean text
    print("\n=== CLEAN TEXT INFO ===")  
    print(f"Total length: {len(clean_text)} characters")
    debug_print(clean_text[:200])

    # Get word positions
    print("\nGetting word positions...")
    word_positions = get_word_positions(clean_text, output_file)

    # Print statistics
    if word_positions:
        print("\n=== STATISTICS ===")
        print(f"Total words found: {len(word_positions)}")

        # Print first 10 words for verification
        print("\nFirst 10 words found:")
        for i, word in enumerate(word_positions[:10], 1):
            print(f"{i}. '{word['text']}' at {word['start']}-{word['end']}")

    print(f"\nSaved word positions to {output_file}")

if __name__ == "__main__":
    main()
