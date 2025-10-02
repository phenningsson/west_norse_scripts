import json

def debug_print(text, highlight_start=None, highlight_end=None):
    """Prints text with optional highlighted section"""
    if highlight_start is not None and highlight_end is not None:
        # Print the context before the match
        context_before = text[max(0, highlight_start-20):highlight_start]
        print(context_before, end='')

        # Print the matched portion in color (using ANSI escape codes)
        matched = text[highlight_start:highlight_end+1]
        print(f"\033[91m{matched}\033[0m", end='')  # red text for the match

        # Print the context after the match
        context_after = text[highlight_end+1:highlight_end+21]
        print(context_after)

        # Print position markers
        print(' ' * len(context_before) + '^' * len(matched))
    else:
        print(text)

def verify_positions(clean_text, entity_positions):
    """Verify and display each entity's position with context"""
    print("\n=== VERIFICATION REPORT ===\n")

    for i, entity in enumerate(entity_positions, 1):
        start = entity['start']
        end = entity['end']
        length = end - start + 1
        matched_text = clean_text[start:end+1]

        print(f"\nEntity {i}: '{entity['text']}' ({entity['label']})")
        print(f"Reported positions: start={start}, end={end} (length={length})")
        print("Actual matched text length:", len(matched_text))

        if matched_text != entity['text']:
            print(f"WARNING: Mismatch between reported text and actual text in document!")
            print(f"Reported: '{entity['text']}'")
            print(f"Actual:   '{matched_text}'")
        else:
            print("Text matches perfectly!")

        print("\nContext with match highlighted:")
        debug_print(clean_text[max(0, start-30):min(len(clean_text), end+31)],
                   start-max(0, start-30) if start > 30 else start,
                   end-max(0, start-30) if start > 30 else end)

        print("\nDetailed position info:")
        print(f"Start character: '{clean_text[start]}' (position {start})")
        print(f"End character:   '{clean_text[end]}' (position {end})")
        print(f"Text length: {len(matched_text)} (calculated as end-start+1={end}-{start}+1={length})")

        # Check if positions make sense with the text
        if matched_text != clean_text[start:end+1]:
            print("ERROR: Positions don't match the text!")
        else:
            print("Positions verify correctly with the text.")

# Load clean text
with open('am_162_B_0_fol_cleaned.txt', 'r', encoding='utf-8') as file:
    clean_text = file.read()

# Load the entity positions 
with open('am162b0_entities_with_positions.json', 'r', encoding='utf-8') as file:
    entity_positions = json.load(file)

# Verify the positions
verify_positions(clean_text, entity_positions)
