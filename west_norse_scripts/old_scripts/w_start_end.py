import json
import re
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

def extract_word_boundaries(clean_text):
    """
    Extracts word boundaries from the text and returns them in a structured format.
    Also saves the results to a JSON file.

    Args:
        clean_text: The text to analyze

    Returns:
        List of dictionaries, each containing word text and start/end positions
    """
    # This pattern matches sequences of non-whitespace characters
    word_pattern = re.compile(r'\S+')
    words_data = []

    for match in word_pattern.finditer(clean_text):
        start = match.start()
        end = match.end() - 1  # Convert to inclusive end position
        word = match.group()

        words_data.append({
            'text': word,
            'start': start,
            'end': end
        })

    return words_data

def main():
    # Load the clean text
    input_file = 'olafs_saga_helga/olafs_saga_helga_clean.txt'
    output_file = 'olafs_word_boundaries.json'

    with open(input_file, 'r', encoding='utf-8') as file:
        clean_text = file.read()

    # Prints info about the clean text
    print("=== CLEAN TEXT INFO ===")
    print(f"Total length: {len(clean_text)} characters")
    debug_print(clean_text[:100])  # Print first 100 chars with debug representation

    # Extract and save word boundaries
    words_data = extract_word_boundaries(clean_text)

    # Save word boundaries to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(words_data, f, ensure_ascii=False, indent=2)

    print(f"\nWord boundaries saved to {output_file}")
    print(f"Found {len(words_data)} words")

    # For verification, print first 10 and last 10 words with their positions
    print("\nFirst 10 words:")
    for i, word_data in enumerate(words_data[:10], 1):
        print(f"{i}. '{word_data['text']}' at positions {word_data['start']}-{word_data['end']}")

    if len(words_data) > 10:
        print("\nLast 10 words:")
        for i, word_data in enumerate(words_data[-10:], len(words_data)-9):
            print(f"{i}. '{word_data['text']}' at positions {word_data['start']}-{word_data['end']}")

if __name__ == "__main__":
    main()
