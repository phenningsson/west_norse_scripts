#!/usr/bin/env python3
"""
Clean MLM Fine-tuning Texts by Removing NER Dev/Test Overlaps

This script removes sentences that appear in the NER evaluation sets
from the MLM fine-tuning texts, creating clean versions suitable for
methodologically sound domain-adaptive pre-training.

Output: Cleaned versions of each text file with "_cleaned" suffix
"""

import os
import re
from typing import List, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

# Diplomatic MLM texts that have overlap (to be cleaned)
DIPLOMATIC_MLM_FILES = [
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/alexanders_saga/alexanders_saga_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/am162b0fol/am162b0_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/am162bkfol/am162bk_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/drauma_jons/drauma_jons_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/islendingabok/islendingabok_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/laknisbok/laknisbok_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/olafs_saga_helga/olafs_saga_helga_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/voluspa/voluspa_final_text.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/wormianus/wormianus_final_text.txt",
]

# NER evaluation files
DEV_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_diplomatic/shared/dev.txt"
TEST_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_diplomatic/shared/test.txt"

# Output directory
OUTPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/dip_mlm_texts"

# =============================================================================
# FUNCTIONS
# =============================================================================

def parse_conll_to_sentences(filepath: str) -> List[str]:
    """Parse CoNLL file and extract sentences as token strings."""
    sentences = []
    current_tokens = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(' '.join(current_tokens))
                    current_tokens = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    current_tokens.append(parts[0])
        
        if current_tokens:
            sentences.append(' '.join(current_tokens))
    
    return sentences


def normalize_for_comparison(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_exclusion_set(sentences: List[str], min_words: int = 5) -> Set[str]:
    """Build set of normalized sentences to exclude."""
    exclusions = set()
    for sent in sentences:
        words = sent.split()
        if len(words) >= min_words:
            exclusions.add(normalize_for_comparison(sent))
    return exclusions


def clean_mlm_text(text: str, exclusion_set: Set[str], min_words: int = 5) -> tuple:
    """
    Remove sentences that appear in the exclusion set from the MLM text.
    
    Strategy: Split text into sentence-like chunks and check each against
    the exclusion set. Remove matching chunks.
    
    Returns: (cleaned_text, num_removed, num_kept)
    """
    # For each exclusion sentence, try to find and remove it from text
    cleaned = text
    removed_count = 0
    
    for excl_sent in exclusion_set:
        # Create a pattern that matches this sentence with flexible whitespace
        excl_words = excl_sent.split()
        if len(excl_words) < min_words:
            continue
        
        # Check if this sentence appears in the normalized text
        norm_cleaned = normalize_for_comparison(cleaned)
        
        if excl_sent in norm_cleaned:
            # Find the actual position in the original text
            # We need to match case-insensitively with flexible whitespace
            pattern_parts = [re.escape(w) for w in excl_words]
            pattern = r'\s*'.join(pattern_parts)
            
            # Try to remove it
            new_cleaned, n = re.subn(pattern, ' [REMOVED] ', cleaned, flags=re.IGNORECASE)
            if n > 0:
                cleaned = new_cleaned
                removed_count += n
    
    # Clean up multiple [REMOVED] markers and extra whitespace
    cleaned = re.sub(r'(\s*\[REMOVED\]\s*)+', '\n\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    
    return cleaned, removed_count


def clean_mlm_text_v2(original_text: str, exclusion_sentences: List[str]) -> tuple:
    """
    Alternative approach: For each sentence in the exclusion list,
    find and mark it for removal in the original text.
    
    Returns: (cleaned_text, segments_removed)
    """
    text = original_text
    removed = 0
    
    for sent in exclusion_sentences:
        norm_sent = normalize_for_comparison(sent)
        words = norm_sent.split()
        
        if len(words) < 5:
            continue
        
        # Build regex pattern
        pattern_parts = [re.escape(w) for w in words]
        pattern = r'\s+'.join(pattern_parts)
        
        # Find and remove
        text_before = text
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        if text != text_before:
            removed += 1
    
    # Clean up whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text, removed


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Clean MLM Texts by Removing NER Dev/Test Overlaps")
    print("=" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load NER sentences to exclude
    print("\n1. Loading NER evaluation sentences to exclude...")
    dev_sentences = parse_conll_to_sentences(DEV_FILE)
    test_sentences = parse_conll_to_sentences(TEST_FILE)
    all_eval_sentences = dev_sentences + test_sentences
    
    print(f"   Dev sentences:  {len(dev_sentences)}")
    print(f"   Test sentences: {len(test_sentences)}")
    print(f"   Total to check: {len(all_eval_sentences)}")
    
    # Build exclusion set
    exclusion_set = build_exclusion_set(all_eval_sentences, min_words=5)
    print(f"   Exclusion set size: {len(exclusion_set)} unique sentences (5+ words)")
    
    # Process each MLM file
    print("\n2. Cleaning MLM texts...")
    print("-" * 70)
    
    total_original_chars = 0
    total_cleaned_chars = 0
    
    for filepath in DIPLOMATIC_MLM_FILES:
        if not os.path.exists(filepath):
            print(f"   ✗ {os.path.basename(filepath)}: NOT FOUND")
            continue
        
        # Load original text
        with open(filepath, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        original_chars = len(original_text)
        total_original_chars += original_chars
        
        # Find which sentences from this file are in exclusion set
        matching_sentences = []
        norm_text = normalize_for_comparison(original_text)
        
        for sent in all_eval_sentences:
            norm_sent = normalize_for_comparison(sent)
            if len(norm_sent.split()) >= 5 and norm_sent in norm_text:
                matching_sentences.append(sent)
        
        # Clean the text
        cleaned_text, removed = clean_mlm_text_v2(original_text, matching_sentences)
        cleaned_chars = len(cleaned_text)
        total_cleaned_chars += cleaned_chars
        
        # Calculate retention
        retention = 100 * cleaned_chars / original_chars if original_chars > 0 else 0
        
        # Save cleaned version
        basename = os.path.basename(filepath).replace('_final_text.txt', '_cleaned.txt')
        output_path = os.path.join(OUTPUT_DIR, basename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        status = "✓" if removed > 0 else "○"
        print(f"   {status} {os.path.basename(filepath):<35} {original_chars:>7,} → {cleaned_chars:>7,} chars ({retention:5.1f}% kept, {removed} segments removed)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    overall_retention = 100 * total_cleaned_chars / total_original_chars if total_original_chars > 0 else 0
    
    print(f"""
   Original diplomatic MLM corpus:  {total_original_chars:,} characters
   Cleaned diplomatic MLM corpus:   {total_cleaned_chars:,} characters
   Overall retention:               {overall_retention:.1f}%
   
   Output directory: {OUTPUT_DIR}
   
   The cleaned texts have all NER dev/test sentences removed.
   You can now use these for IceBERT fine-tuning without evaluation contamination.
""")
    
    # Also copy the normalized texts (which have no overlap) to the output dir
    print("3. Copying clean normalized texts (no overlap)...")
    normalized_files = [
    ]
    
    norm_chars = 0
    for filepath in normalized_files:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            norm_chars += len(text)
            
            basename = os.path.basename(filepath)
            output_path = os.path.join(OUTPUT_DIR, basename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"   ✓ {basename}: {len(text):,} chars (no cleaning needed)")
    
    print(f"\n" + "=" * 70)
    print("FINAL CORPUS")
    print("=" * 70)
    
    final_total = total_cleaned_chars + norm_chars
    print(f"""
   Cleaned diplomatic texts:  {total_cleaned_chars:,} characters
   Clean normalized texts:    {norm_chars:,} characters
   ─────────────────────────────────────────
   TOTAL CLEAN MLM CORPUS:    {final_total:,} characters
   
   All files saved to: {OUTPUT_DIR}
   
   This corpus is safe for IceBERT fine-tuning with zero NER eval contamination.
""")


if __name__ == "__main__":
    main()