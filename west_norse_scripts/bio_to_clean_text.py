#!/usr/bin/env python3
"""
Convert BIO-tagged corpus files to clean text.

This script removes BIO tags and reconstructs the original text,
preserving sentence and paragraph boundaries.

Input format:
    Illmennska	O
    Holbergs	B-Person
    er	O
    ...

Output format:
    Illmennska Holbergs er ...
"""

from pathlib import Path


# ============================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================

# Input directory containing the BIO-tagged .txt files
INPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/bio_entities"

# Output directory for clean text files
OUTPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/clean_texts"

# ============================================================


def convert_bio_to_text(filepath: str) -> str:
    """
    Convert a BIO-tagged file to clean text.
    
    Args:
        filepath: Path to the BIO-tagged file
        
    Returns:
        Clean text with sentences on separate lines (no empty lines between)
    """
    sentences = []
    current_tokens = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            # Empty line = end of sentence
            if not line.strip():
                if current_tokens:
                    sentences.append(' '.join(current_tokens))
                    current_tokens = []
                continue
            
            # Extract token (first column before tab)
            parts = line.split('\t')
            if parts:
                token = parts[0]
                current_tokens.append(token)
    
    # Don't forget last sentence
    if current_tokens:
        sentences.append(' '.join(current_tokens))
    
    return '\n'.join(sentences)


def process_directory(input_dir: str, output_dir: str):
    """
    Process all .txt files in input directory.
    
    Args:
        input_dir: Directory containing BIO-tagged .txt files
        output_dir: Directory for output clean text files
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    txt_files = sorted(Path(input_dir).glob('*.txt'))
    
    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return
    
    print(f"Found {len(txt_files)} files to process...")
    print("-" * 50)
    
    total_sentences = 0
    total_tokens = 0
    
    for filepath in txt_files:
        text = convert_bio_to_text(str(filepath))
        
        # Count stats
        lines = [l for l in text.split('\n') if l.strip()]
        num_sentences = len(lines)
        num_tokens = sum(len(l.split()) for l in lines)
        total_sentences += num_sentences
        total_tokens += num_tokens
        
        # Write output
        output_path = Path(output_dir) / filepath.name
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"  {filepath.name}: {num_sentences} sentences, {num_tokens:,} tokens")
    
    print("-" * 50)
    print(f"\nTotal: {total_sentences:,} sentences, {total_tokens:,} tokens")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    process_directory(INPUT_DIR, OUTPUT_DIR)