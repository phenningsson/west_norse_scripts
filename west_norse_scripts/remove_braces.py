"""
Clean text file by removing {curly braces} and content inside them
"""

import re

# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = 'egil_saga/egil_saga_am132_norm.txt'      # Your input file
OUTPUT_FILE = 'egil_saga/egil_saga_am132_new.txt'  # Cleaned output file

# ============================================================================
# MAIN
# ============================================================================

print("="*70)
print("REMOVING CURLY BRACES FROM TEXT")
print("="*70)

# Read input file
print(f"\nReading: {INPUT_FILE}")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Clean each line
cleaned_lines = []
changes_count = 0

for line in lines:
    original = line.strip()
    
    if not original:  # Skip empty lines
        continue
    
    # Remove {anything} pattern
    cleaned = re.sub(r'\{[^}]*\}', '', original)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces → single space
    cleaned = cleaned.strip()  # Remove leading/trailing spaces
    
    # Track changes
    if cleaned != original:
        changes_count += 1
    
    # Only keep non-empty lines
    if cleaned:
        cleaned_lines.append(cleaned)

# Save cleaned file
print(f"\nSaving cleaned file: {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned_lines))

# Statistics
print("\n" + "="*70)
print("CLEANING COMPLETE")
print("="*70)
print(f"Original lines:     {len(lines)}")
print(f"Cleaned lines:      {len(cleaned_lines)}")
print(f"Lines modified:     {changes_count}")
print(f"Lines removed:      {len(lines) - len(cleaned_lines)}")
print(f"\n✓ Cleaned file saved to: {OUTPUT_FILE}")
print("="*70)

# Show examples of changes
if changes_count > 0:
    print("\nSample of changes (first 5):")
    print("-" * 70)
    
    count = 0
    for i, line in enumerate(lines):
        original = line.strip()
        if not original:
            continue
            
        cleaned = re.sub(r'\{[^}]*\}', '', original)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if cleaned != original and count < 5:
            print(f"\nBEFORE: {original}")
            print(f"AFTER:  {cleaned}")
            count += 1
            
        if count >= 5:
            break