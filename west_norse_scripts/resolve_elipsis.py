import re
import os

def resolve_hyphens(text):
    """
    Resolve hyphens marked with · in Old Icelandic text
    Combines hyphenated words and properly formats line breaks
    """
    # Replace hyphen with combined word followed by line break
    # This pattern now captures and removes any trailing space after the second word part
    resolved_text = re.sub(r'([^\s·]+)·\s*([^\s·]+)\s*', r'\1\2\n', text)
    
    # Clean up any lines that start with whitespace
    # This handles edge cases and ensures no leading spaces
    resolved_text = '\n'.join(line.lstrip() for line in resolved_text.split('\n'))
    
    # Remove any empty lines that might have been created
    resolved_text = '\n'.join(line for line in resolved_text.split('\n') if line.strip())
    
    return resolved_text

def process_file(input_file_path, output_file_path):
    """Process file to resolve hyphens"""
    
    # Check if the input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: The input file '{input_file_path}' does not exist.")
        return

    # Read the input file
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            input_text = file.read()
        print(f"✓ Successfully read: {input_file_path}")
        print(f"  Original length: {len(input_text)} characters")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Count hyphens before processing
    hyphen_count = input_text.count('·')
    print(f"  Hyphens found: {hyphen_count}")

    # Process the text
    output_text = resolve_hyphens(input_text)
    print("✓ Text processing completed")
    print(f"  Processed length: {len(output_text)} characters")

    # Verify no leading spaces on lines
    lines = output_text.split('\n')
    lines_with_leading_space = sum(1 for line in lines if line and line[0] == ' ')
    print(f"  Lines with leading spaces: {lines_with_leading_space}")
    
    if lines_with_leading_space > 0:
        print("  ⚠ Warning: Some lines still have leading spaces!")
    else:
        print("  ✓ All lines properly formatted (no leading spaces)")

    # Print sample of processed text
    print("\nSample of processed text (first 500 chars):")
    print("-" * 70)
    print(output_text[:500])
    print("-" * 70)

    # Write the output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(output_text)
        print(f"\n✓ Successfully saved to: {output_file_path}")
    except Exception as e:
        print(f"Error writing file: {e}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Update these paths to your files
    input_file_path = '/Users/phenningsson/Downloads/west_norse_scripts/viga_glums_saga_dipl_cleaned.txt'  
    output_file_path = '/Users/phenningsson/Downloads/west_norse_scripts/viga_glums_saga_dipl_cleanedd.txt'
    
    print("="*70)
    print("HYPHEN RESOLUTION SCRIPT")
    print("="*70)
    print(f"\nInput:  {input_file_path}")
    print(f"Output: {output_file_path}\n")
    
    process_file(input_file_path, output_file_path)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)