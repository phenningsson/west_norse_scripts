def find_invalid_chars(input_file):
    """
    Scans a text file for invalid characters and shows them in context.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    print(f"Checking file: {input_file}")
    print("=" * 60)

    invalid_chars = []
    lines = content.split('\n')

    for line_num, line in enumerate(lines, 1):
        for char_num, char in enumerate(line, 1):
            code_point = ord(char)

            # Check for various types of invalid characters
            invalid = False
            category = ""

            # Control characters (except whitespace)
            if code_point < 32 and char not in '\t\n\r':
                invalid = True
                category = 'control character'

            # Replacement character (undefined)
            elif code_point == 0xFFFD:
                invalid = True
                category = 'replacement character'

            # Private use characters
            elif 0xE000 <= code_point <= 0xF8FF:
                invalid = True
                category = 'private use character'

            # Other non-printable characters
            elif not char.isprintable():
                invalid = True
                category = 'non-printable character'

            if invalid:
                # Get context around the character (20 chars before and after)
                start = max(0, char_num - 1 - 20)
                end = min(len(line), char_num - 1 + 20)
                context = line[start:end]

                # Highlight the invalid character in context
                highlighted = context.replace(char, f"\033[1;31m{char}\033[0m")

                invalid_chars.append({
                    'line': line_num,
                    'position': char_num,
                    'char': char,
                    'unicode': f'U+{code_point:04X}',
                    'category': category,
                    'context': context,
                    'highlighted': highlighted
                })

    if not invalid_chars:
        print("No invalid characters found.")
        return

    print(f"Found {len(invalid_chars)} invalid characters:")
    print("-" * 80)

    for char in invalid_chars:
        print(f"Line {char['line']}, Position {char['position']}:")
        print(f"  Character: {char['char']!r} ({char['unicode']})")
        print(f"  Category: {char['category']}")
        print(f"  Context: ...{char['highlighted']}...")
        print("-" * 80)

def main():
    input_file = 'egil_saga/egil_saga_am132_norm.txt'
    find_invalid_chars(input_file)

if __name__ == "__main__":
    main()
