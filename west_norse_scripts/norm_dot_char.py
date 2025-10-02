import unicodedata
import re

def normalize_unclear_chars(text):
    """
    Normalizes text by removing dots below characters that indicate unclear readings.
    Handles both pre-composed characters with dots and sequences with combining dots.
    Preserves all other characters and punctuation.

    Args:
        text: Input text with unclear characters marked with dots below

    Returns:
        Normalized text with unclear markings removed
    """
    # Create a mapping of characters with dots below to their normalized versions
    # This handles both pre-composed characters and sequences with combining dots
    replacement_map = {
        'ạ': 'a', 'ḅ': 'b', 'c̣': 'c', 'ḍ': 'd', 'ẹ': 'e',
        'f̣': 'f', 'g̣': 'g', 'ḥ': 'h', 'ị': 'i', 'j̣': 'j',
        'ḳ': 'k', 'ḷ': 'l', 'ṃ': 'm', 'ṇ': 'n', 'ọ': 'o',
        'p̣': 'p', 'q̣': 'q', 'ṛ': 'r', 'ṣ': 's', 'ṭ': 't',
        'ụ': 'u', 'ṿ': 'v', 'ẉ': 'w', 'x̣': 'x', 'ỵ': 'y',
        'ẓ': 'z', 'þ̣': 'þ', 'ð̣': 'ð', 'æ̣': 'æ', 'ọ̈': 'ö',
        'Ạ': 'A', 'Ḅ': 'B', 'C̣': 'C', 'Ḍ': 'D', 'Ẹ': 'E',
        'F̣': 'F', 'G̣': 'G', 'Ḥ': 'H', 'Ị': 'I', 'J̣': 'J',
        'Ḳ': 'K', 'Ḷ': 'L', 'Ṃ': 'M', 'Ṇ': 'N', 'Ọ': 'O',
        'P̣': 'P', 'Q̣': 'Q', 'Ṛ': 'R', 'Ṣ': 'S', 'Ṭ': 'T',
        'Ụ': 'U', 'Ṿ': 'V', 'Ẉ': 'W', 'X̣': 'X', 'Ỵ': 'Y',
        'Ẓ': 'Z', 'Þ̣': 'Þ', 'Ð̣': 'Ð', 'Æ̣': 'Æ', 'Ọ̈': 'Ö',
        # Handle combining dot below (U+0323)
        '\u0323': ''  # This is the combining dot below character
    }

    # Process the text in two steps:
    # 1. First handle pre-composed characters with dots
    for dotted, clean in replacement_map.items():
        if len(dotted) == 2:  # These are base char + combining dot
            # Find all occurrences of the sequence
            text = text.replace(dotted, clean)
        else:
            # Find all occurrences of the character
            text = text.replace(dotted, clean)

    # 2. Handle any remaining combining dots (they might appear alone)
    # This is more efficient than using regex for simple replacements
    text = text.replace('\u0323', '')

    return text

def process_file(input_file, output_file):
    """
    Processes a file to normalize unclear characters.

    Args:
        input_file: Path to input text file
        output_file: Path to save normalized text file
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            normalized_line = normalize_unclear_chars(line)
            f_out.write(normalized_line)

def main():
    input_file = 'alexanders_saga_clean.txt' 
    output_file = 'alexanders_saga_norm.txt'  
    print(f"Processing {input_file}...")
    process_file(input_file, output_file)
    print(f"Normalized text saved to {output_file}")

if __name__ == "__main__":
    main()
