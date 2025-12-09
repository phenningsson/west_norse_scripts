#!/usr/bin/env python3
"""
Create Training Set from Menota Data (Excluding Dev/Test)

This script creates a training set from Menota CoNLL files while ensuring
NO sentences from the existing dev/test splits are included.

This allows you to add a new experiment without redoing your entire split.

Usage:
    python create_menota_train_excluding_devtest.py

Output:
    menota_train.txt - Training data with dev/test sentences removed
"""

import os
from typing import List, Tuple, Set
from collections import defaultdict

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Menota source files (your uploaded files)
MENOTA_FILES = [
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/menota_normalised/alexander_saga_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/menota_normalised/am162b0_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/menota_normalised/am162bk_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/menota_normalised/drauma_jons_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/menota_normalised/islendingabok_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/menota_normalised/laknisbok_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/menota_normalised/olafs_saga_conll.txt",
]

# Your existing dev/test files
DEV_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_normalised/shared/dev.txt"
TEST_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_normalised/shared/test.txt"

# Output file
OUTPUT_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_normalised/menota_only/menota_only.txt"

# =============================================================================
# PARSING FUNCTIONS
# =============================================================================

def parse_conll_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """Parse a CoNLL file into sentences (list of (token, label) tuples)."""
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    token, label = parts[0], parts[-1]
                    current_sentence.append((token, label))
        
        if current_sentence:
            sentences.append(current_sentence)
    
    return sentences


def sentence_to_key(sentence: List[Tuple[str, str]]) -> str:
    """
    Convert a sentence to a hashable string key for comparison.
    Uses both tokens and labels to ensure exact match.
    """
    return "|||".join(f"{token}\t{label}" for token, label in sentence)


def write_conll_file(filepath: str, sentences: List[List[Tuple[str, str]]]):
    """Write sentences to a CoNLL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            if i < len(sentences) - 1:
                f.write('\n')


def count_entities(sentences: List[List[Tuple[str, str]]]) -> dict:
    """Count entities by type in a list of sentences."""
    counts = defaultdict(int)
    for sentence in sentences:
        for token, label in sentence:
            if label.startswith('B-'):
                counts[label[2:]] += 1
    return dict(counts)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Create Menota Training Set (Excluding Dev/Test)")
    print("=" * 70)
    
    # =========================================================================
    # 1. Load Dev and Test sentences to exclude
    # =========================================================================
    print("\n1. Loading Dev/Test sentences to exclude...")
    
    excluded_keys: Set[str] = set()
    
    # Load dev
    if os.path.exists(DEV_FILE):
        dev_sentences = parse_conll_file(DEV_FILE)
        for sent in dev_sentences:
            excluded_keys.add(sentence_to_key(sent))
        print(f"   Dev: {len(dev_sentences)} sentences loaded")
    else:
        print(f"   WARNING: Dev file not found: {DEV_FILE}")
        dev_sentences = []
    
    # Load test
    if os.path.exists(TEST_FILE):
        test_sentences = parse_conll_file(TEST_FILE)
        for sent in test_sentences:
            excluded_keys.add(sentence_to_key(sent))
        print(f"   Test: {len(test_sentences)} sentences loaded")
    else:
        print(f"   WARNING: Test file not found: {TEST_FILE}")
        test_sentences = []
    
    print(f"\n   Total unique sentences to exclude: {len(excluded_keys)}")
    
    # =========================================================================
    # 2. Load Menota files and filter out dev/test sentences
    # =========================================================================
    print("\n2. Loading Menota files and filtering...")
    
    train_sentences = []
    stats_per_file = {}
    
    for filepath in MENOTA_FILES:
        if not os.path.exists(filepath):
            print(f"   ✗ {os.path.basename(filepath)}: NOT FOUND")
            continue
        
        file_sentences = parse_conll_file(filepath)
        original_count = len(file_sentences)
        
        # Filter out sentences that are in dev/test
        filtered = []
        excluded_count = 0
        for sent in file_sentences:
            key = sentence_to_key(sent)
            if key not in excluded_keys:
                filtered.append(sent)
            else:
                excluded_count += 1
        
        train_sentences.extend(filtered)
        
        stats_per_file[os.path.basename(filepath)] = {
            'original': original_count,
            'kept': len(filtered),
            'excluded': excluded_count,
        }
        
        print(f"   ✓ {os.path.basename(filepath)}: {original_count} → {len(filtered)} "
              f"(excluded {excluded_count} in dev/test)")
    
    # =========================================================================
    # 3. Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. Summary")
    print("=" * 70)
    
    total_original = sum(s['original'] for s in stats_per_file.values())
    total_kept = sum(s['kept'] for s in stats_per_file.values())
    total_excluded = sum(s['excluded'] for s in stats_per_file.values())
    
    print(f"\n   Total Menota sentences: {total_original}")
    print(f"   Excluded (in dev/test): {total_excluded}")
    print(f"   Kept for training:      {total_kept}")
    
    # Entity counts
    train_entities = count_entities(train_sentences)
    dev_entities = count_entities(dev_sentences) if dev_sentences else {}
    test_entities = count_entities(test_sentences) if test_sentences else {}
    
    print(f"\n   Entity distribution in NEW training set:")
    for etype, count in sorted(train_entities.items()):
        print(f"     {etype}: {count}")
    
    print(f"\n   Entity distribution in Dev set:")
    for etype, count in sorted(dev_entities.items()):
        print(f"     {etype}: {count}")
    
    print(f"\n   Entity distribution in Test set:")
    for etype, count in sorted(test_entities.items()):
        print(f"     {etype}: {count}")
    
    # =========================================================================
    # 4. Write output file
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. Writing output file")
    print("=" * 70)
    
    write_conll_file(OUTPUT_FILE, train_sentences)
    print(f"\n   Written: {OUTPUT_FILE}")
    print(f"   Sentences: {len(train_sentences)}")
    
    # =========================================================================
    # 5. Verification
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. Verification (checking for data leakage)")
    print("=" * 70)
    
    # Re-read the output file and verify no overlap
    output_sentences = parse_conll_file(OUTPUT_FILE)
    output_keys = set(sentence_to_key(s) for s in output_sentences)
    
    overlap_with_dev = output_keys & set(sentence_to_key(s) for s in dev_sentences)
    overlap_with_test = output_keys & set(sentence_to_key(s) for s in test_sentences)
    
    if overlap_with_dev:
        print(f"   ⚠ WARNING: {len(overlap_with_dev)} sentences overlap with dev!")
    else:
        print(f"   ✓ No overlap with dev set")
    
    if overlap_with_test:
        print(f"   ⚠ WARNING: {len(overlap_with_test)} sentences overlap with test!")
    else:
        print(f"   ✓ No overlap with test set")
    
    # =========================================================================
    # Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ NEW MENOTA TRAINING SET                                             │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Output file: {OUTPUT_FILE:<52} │
  │ Sentences:   {len(train_sentences):<52} │
  │ Entities:    Person={train_entities.get('Person', 0)}, Location={train_entities.get('Location', 0):<30} │
  ├─────────────────────────────────────────────────────────────────────┤
  │ EXISTING DEV/TEST (unchanged)                                       │
  │ Dev:  {len(dev_sentences)} sentences                                              │
  │ Test: {len(test_sentences)} sentences                                             │
  └─────────────────────────────────────────────────────────────────────┘

  This training set is SAFE to use with your existing dev/test splits.
  No data leakage - all dev/test sentences have been excluded.

  USAGE:
    TRAIN_FILE = "{OUTPUT_FILE}"
    DEV_FILE   = "{DEV_FILE}"
    TEST_FILE  = "{TEST_FILE}"
""")


if __name__ == "__main__":
    main()