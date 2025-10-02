import json
from termcolor import colored

def debug_print(text, highlight_start=None, highlight_end=None):
    """Prints text with optional highlighted section"""
    if highlight_start is not None and highlight_end is not None:
        # Print the context before the match
        context_before = text[max(0, highlight_start-20):highlight_start]
        print(context_before, end='')

        # Print the matched portion in color
        matched = text[highlight_start:highlight_end+1]
        print(colored(matched, 'red', attrs=['bold']), end='')

        # Print the context after the match
        context_after = text[highlight_end+1:highlight_end+21]
        print(context_after)

        # Print position markers
        print(' ' * len(context_before) + '^' * len(matched))
    else:
        print(text)

def verify_word_boundaries(clean_text, word_boundaries):
    """
    Verify that the word boundaries correctly match the text.

    Args:
        clean_text: The full text
        word_boundaries: List of word boundary dictionaries (from JSON file)
    """
    print("\n=== WORD BOUNDARY VERIFICATION REPORT ===\n")
    print(f"Total words to verify: {len(word_boundaries)}")

    # Extract word positions from the text itself to compare
    import re
    expected_words = []
    expected_starts = []
    expected_ends = []

    word_pattern = re.compile(r'\S+')
    for match in word_pattern.finditer(clean_text):
        expected_starts.append(match.start())
        expected_ends.append(match.end() - 1)  # Convert to inclusive end position
        expected_words.append(match.group())

    # Check if word counts match
    if len(word_boundaries) != len(expected_words):
        print(colored(f"\nERROR: Word count mismatch! Reported {len(word_boundaries)} words, but found {len(expected_words)} in text.", 'red'))
        return False

    # Check each word position
    all_correct = True
    for i, word_data in enumerate(word_boundaries):
        actual_text = clean_text[word_data['start']:word_data['end']+1]

        if i < len(expected_words):
            expected_word = expected_words[i]
            expected_start = expected_starts[i]
            expected_end = expected_ends[i]

            if (actual_text != word_data['text'] or
                word_data['start'] != expected_start or
                word_data['end'] != expected_end):
                all_correct = False
                print(colored(f"\nIssue found with word {i+1}: '{word_data['text']}'", 'red'))
                print(f"Reported positions: start={word_data['start']}, end={word_data['end']}")
                print(f"Expected positions: start={expected_start}, end={expected_end}")

                if actual_text != word_data['text']:
                    print(colored("Text mismatch!", 'red'))
                    print(f"Reported text: '{word_data['text']}'")
                    print(f"Actual extracted text: '{actual_text}'")
                    print(f"Expected word: '{expected_word}'")

                print("\nContext:")
                debug_print(
                    clean_text[max(0, expected_start-20):min(len(clean_text), expected_end+21)],
                    expected_start-max(0, expected_start-20) if expected_start > 20 else expected_start,
                    expected_end-max(0, expected_start-20) if expected_start > 20 else expected_end
                )

                # Show the reported position if different
                if (word_data['start'] != expected_start or
                    word_data['end'] != expected_end):
                    print(colored("\nReported position shown below:", 'yellow'))
                    debug_print(
                        clean_text[max(0, word_data['start']-20):min(len(clean_text), word_data['end']+21)],
                        word_data['start']-max(0, word_data['start']-20) if word_data['start'] > 20 else word_data['start'],
                        word_data['end']-max(0, word_data['start']-20) if word_data['start'] > 20 else word_data['end']
                    )

    # If everything matched
    if all_correct:
        print(colored("\nAll word boundaries verified correctly!", 'green'))
        print(f"Successfully verified {len(word_boundaries)} words.")

    # Sample some correct words too
    sample_indices = [0, len(word_boundaries)//4, len(word_boundaries)//2, len(word_boundaries)-1]
    for idx in sample_indices:
        if idx < len(word_boundaries):
            word_data = word_boundaries[idx]
            expected_start = expected_starts[idx]
            expected_end = expected_ends[idx]

            # If we had issues above, skip the samples
            if not all_correct and (word_data['start'] != expected_start or word_data['end'] != expected_end):
                continue

            print(f"\nSample word {idx+1}: '{word_data['text']}'")
            print(f"Positions: start={word_data['start']}, end={word_data['end']}")
            debug_print(
                clean_text[max(0, word_data['start']-20):min(len(clean_text), word_data['end']+21)],
                word_data['start']-max(0, word_data['start']-20) if word_data['start'] > 20 else word_data['start'],
                word_data['end']-max(0, word_data['start']-20) if word_data['start'] > 20 else word_data['end']
            )

    return all_correct

def main():
    # Load the clean text
    text_file = 'am_162_B_0_fol_cleaned.txt'
    word_boundaries_file = 'am162b0_word_boundaries.json'

    print(f"Loading text from {text_file}...")
    with open(text_file, 'r', encoding='utf-8') as file:
        clean_text = file.read()

    print(f"\nLoading word boundaries from {word_boundaries_file}...")
    with open(word_boundaries_file, 'r', encoding='utf-8') as file:
        word_boundaries = json.load(file)

    # Verify the word boundaries
    verify_word_boundaries(clean_text, word_boundaries)

if __name__ == "__main__":
    main()
