import re

def remove_line_numbers_and_brackets(input_file, output_file):
    """
    Removes line numbers from the beginning of each line and removes square brackets
    while preserving their contents. Saves the cleaned text to the output file.

    Args:
        input_file: Path to input text file
        output_file: Path to save cleaned text file
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            # Remove leading numbers (with optional punctuation) and any following whitespace
            line = re.sub(r'^\d+[^\w\s]*\s*', '', line)
            # Remove all square brackets, keeping their contents
            cleaned_line = re.sub(r'[\[\]]', '', line)
            f_out.write(cleaned_line)

def main():
    input_file = '/Users/phenningsson/Downloads/west_norse_scripts/viga_glums_saga_dipl.txt'
    output_file = '/Users/phenningsson/Downloads/west_norse_scripts/viga_glums_saga_dipl_cleaned.txt' 
    print(f"Processing {input_file}...")
    remove_line_numbers_and_brackets(input_file, output_file)
    print(f"Cleaned text saved to {output_file}")

if __name__ == "__main__":
    main()
