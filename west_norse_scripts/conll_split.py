#!/usr/bin/env python3
"""
Domain-Specific Data Splitting for NER Training

Creates train/dev/test splits where:
- Train: All data (Old Icelandic + Modern Icelandic MIM-GOLD-NER)
- Dev: Old Icelandic only (for tuning toward target domain)
- Test: Old Icelandic only (for realistic target-domain evaluation)

Edit the CONFIGURATION section below to set your paths.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter


# =============================================================================
# CONFIGURATION -
# =============================================================================

# Old Norse/Icelandic CoNLL files (target domain)
OLD_NORSE_FILES = [
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/1210_jartein_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/1210_thorlakur_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/1250_sturlunga_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/1250_thetubrot_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/1260_jomsvikingar_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/1275_morkin_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/alexander_saga_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/am162b0_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/am162bk_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/drauma_jons_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/islendingabok_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/laknisbok_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format/old_icelandic_all/olafs_saga_conll.txt"
]

# MIM-GOLD-NER filtered files directory (or list specific files)
MIM_GOLD_DIR = '/Users/phenningsson/Downloads/west_norse_scripts/conll_format/mim_gold_ner/filtered'

# Or specify files directly (set to None to use all .txt files in MIM_GOLD_DIR)
MIM_GOLD_FILES = None  # Will use all .txt files in MIM_GOLD_DIR

# Output directory for split files
OUTPUT_DIR = '/Users/phenningsson/Downloads/west_norse_scripts/conll_splits'

# Split ratios for Old Norse data
DEV_RATIO = 0.10   # 10% of Old Norse for dev
TEST_RATIO = 0.10  # 10% of Old Norse for test
# Remaining 80% of Old Norse goes to train (along with ALL MIM-GOLD-NER)

# Random seed for reproducibility
SEED = 42

# =============================================================================
# END CONFIGURATION
# =============================================================================


def read_conll_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """Read CoNLL file and return list of sentences."""
    sentences = []
    current_sentence = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    current_sentence.append((parts[0], parts[1]))
    
    if current_sentence:
        sentences.append(current_sentence)
    
    return sentences


def write_conll_file(filepath: str, sentences: List[List[Tuple[str, str]]]):
    """Write sentences to CoNLL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f'{token}\t{label}\n')
            f.write('\n')


def get_statistics(sentences: List[List[Tuple[str, str]]]) -> Dict:
    """Get statistics for a dataset."""
    entity_counts = Counter()
    total_tokens = 0
    
    for sentence in sentences:
        for token, label in sentence:
            total_tokens += 1
            if label.startswith('B-'):
                entity_counts[label[2:]] += 1
    
    return {
        'sentences': len(sentences),
        'tokens': total_tokens,
        'entities': dict(entity_counts),
        'total_entities': sum(entity_counts.values())
    }


def main():
    random.seed(SEED)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Domain-Specific Data Splitting")
    print("=" * 70)
    print(f"\nStrategy:")
    print(f"  - Test set: Old Icelandic ONLY ({TEST_RATIO*100:.0f}% of Old Icelandic data)")
    print(f"  - Dev set: Old Icelandic ONLY ({DEV_RATIO*100:.0f}% of Old Icelandic data)")
    print(f"  - Train set: Remaining Old Icelandic + ALL MIM-GOLD-NER")
    
    # Load Old Norse data
    print(f"\n{'='*70}")
    print("Loading Old Norse/Icelandic data...")
    print("="*70)
    
    old_norse_sentences = []
    
    for filepath in OLD_NORSE_FILES:
        if os.path.exists(filepath):
            sentences = read_conll_file(filepath)
            old_norse_sentences.extend(sentences)
            filename = os.path.basename(filepath)
            stats = get_statistics(sentences)
            print(f"  {filename}: {stats['sentences']} sentences, {stats['total_entities']} entities")
        else:
            print(f"  Warning: File not found: {filepath}")
    
    print(f"\n  Total Old Norse: {len(old_norse_sentences)} sentences")
    
    # Load MIM-GOLD-NER data
    print(f"\n{'='*70}")
    print("Loading MIM-GOLD-NER data...")
    print("="*70)
    
    mim_sentences = []
    
    if MIM_GOLD_FILES:
        mim_files = MIM_GOLD_FILES
    else:
        mim_gold_path = Path(MIM_GOLD_DIR)
        mim_files = sorted(mim_gold_path.glob('*.txt'))
    
    for filepath in mim_files:
        filepath = str(filepath)
        if os.path.exists(filepath):
            sentences = read_conll_file(filepath)
            mim_sentences.extend(sentences)
            filename = os.path.basename(filepath)
            stats = get_statistics(sentences)
            print(f"  {filename}: {stats['sentences']} sentences, {stats['total_entities']} entities")
        else:
            print(f"  Warning: File not found: {filepath}")
    
    print(f"\n  Total MIM-GOLD-NER: {len(mim_sentences)} sentences")
    
    # Shuffle Old Norse and split into train/dev/test
    print(f"\n{'='*70}")
    print("Creating splits...")
    print("="*70)
    
    random.shuffle(old_norse_sentences)
    
    n_old_norse = len(old_norse_sentences)
    n_test = int(n_old_norse * TEST_RATIO)
    n_dev = int(n_old_norse * DEV_RATIO)
    n_train_old_norse = n_old_norse - n_test - n_dev
    
    # Split Old Norse
    test_sentences = old_norse_sentences[:n_test]
    dev_sentences = old_norse_sentences[n_test:n_test + n_dev]
    train_old_norse = old_norse_sentences[n_test + n_dev:]
    
    # Combine train: Old Norse train portion + ALL MIM-GOLD-NER
    train_sentences = train_old_norse + mim_sentences
    random.shuffle(train_sentences)
    
    # Write files
    write_conll_file(output_path / 'train.txt', train_sentences)
    write_conll_file(output_path / 'dev.txt', dev_sentences)
    write_conll_file(output_path / 'test.txt', test_sentences)
    write_conll_file(output_path / 'train_old_norse_only.txt', train_old_norse)
    
    # Print statistics
    print(f"\n{'='*70}")
    print("Split Statistics")
    print("="*70)
    
    train_stats = get_statistics(train_sentences)
    dev_stats = get_statistics(dev_sentences)
    test_stats = get_statistics(test_sentences)
    train_on_stats = get_statistics(train_old_norse)
    
    print(f"\n  TRAIN (Old Norse + MIM-GOLD-NER):")
    print(f"    Sentences: {train_stats['sentences']:,}")
    print(f"    Tokens: {train_stats['tokens']:,}")
    print(f"    Entities: {train_stats['total_entities']:,}")
    print(f"      - From Old Norse: {train_on_stats['total_entities']:,}")
    print(f"      - From MIM-GOLD-NER: {train_stats['total_entities'] - train_on_stats['total_entities']:,}")
    for etype, count in sorted(train_stats['entities'].items(), key=lambda x: -x[1]):
        print(f"        {etype}: {count:,}")
    
    print(f"\n  DEV (Old Norse ONLY):")
    print(f"    Sentences: {dev_stats['sentences']:,}")
    print(f"    Tokens: {dev_stats['tokens']:,}")
    print(f"    Entities: {dev_stats['total_entities']:,}")
    for etype, count in sorted(dev_stats['entities'].items(), key=lambda x: -x[1]):
        print(f"        {etype}: {count:,}")
    
    print(f"\n  TEST (Old Norse ONLY):")
    print(f"    Sentences: {test_stats['sentences']:,}")
    print(f"    Tokens: {test_stats['tokens']:,}")
    print(f"    Entities: {test_stats['total_entities']:,}")
    for etype, count in sorted(test_stats['entities'].items(), key=lambda x: -x[1]):
        print(f"        {etype}: {count:,}")
    
    # Summary
    print(f"\n{'='*70}")
    print("Output Files")
    print("="*70)
    print(f"""
  Files created in {OUTPUT_DIR}/:
    - train.txt              : Training data (mixed)
    - dev.txt                : Dev data (Old Norse only)
    - test.txt               : Test data (Old Norse only)
    - train_old_norse_only.txt : Old Norse portion of training data
    """)


if __name__ == '__main__':
    main()