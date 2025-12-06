#!/usr/bin/env python3
"""
Diplomatic Old Icelandic NER Experiment Configuration Generator

This script creates a consistent experimental setup for comparing different
NER training approaches on DIPLOMATIC Old Icelandic texts.

Key difference from normalized experiments:
- Dev/Test contain ONLY diplomatic texts (target domain)
- Training can include diplomatic texts, normalized texts, MIM-GOLD-NER, or combinations
- Evaluation always on diplomatic texts for fair comparison

Experiments (base - without resampling):
1. diplo_only:                Diplomatic texts only
2. diplo_plus_norm:           Diplomatic + Normalized Old Icelandic
3. diplo_plus_mim:            Diplomatic + MIM-GOLD-NER (Modern Icelandic)
4. diplo_plus_norm_plus_mim:  Diplomatic + Normalized + MIM-GOLD-NER

Experiments (with resampling):
5. diplo_resamp:                    Diplomatic (resampled)
6. diplo_resamp_plus_norm:          Diplomatic (resampled) + Normalized
7. diplo_resamp_plus_mim:           Diplomatic (resampled) + MIM-GOLD-NER
8. diplo_resamp_plus_norm_plus_mim: Diplomatic (resampled) + Normalized + MIM-GOLD-NER

Research questions answered:
- Does normalized Old Icelandic help? (diplo_plus_norm vs diplo_only)
- Does modern Icelandic help? (diplo_plus_mim vs diplo_only)
- Does the auxiliary data need to be historical? (diplo_plus_norm vs diplo_plus_mim)
- Does combining both help most? (diplo_plus_norm_plus_mim vs others)
- How does resampling interact with data augmentation strategies?

For model comparisons:
- Use same training files with Original IceBERT vs Fine-tuned IceBERT
- This gives you up to 16 total experiments (8 training configs × 2 model versions)

Directory structure created:
    experiments_diplomatic/
    ├── shared/
    │   ├── dev.txt                      # SAME for all experiments (diplomatic only)
    │   └── test.txt                     # SAME for all experiments (diplomatic only)
    ├── diplo_only/
    │   └── train.txt                    # Diplomatic only
    ├── diplo_resamp/
    │   └── train.txt                    # Diplomatic (resampled)
    ├── diplo_plus_norm/
    │   └── train.txt                    # Diplomatic + Normalized Old Icelandic
    ├── diplo_plus_mim/
    │   └── train.txt                    # Diplomatic + MIM-GOLD-NER
    ├── diplo_plus_norm_plus_mim/
    │   └── train.txt                    # Diplomatic + Normalized + MIM-GOLD-NER
    ├── diplo_resamp_plus_norm/
    │   └── train.txt                    # Diplomatic (resampled) + Normalized
    ├── diplo_resamp_plus_mim/
    │   └── train.txt                    # Diplomatic (resampled) + MIM-GOLD-NER
    └── diplo_resamp_plus_norm_plus_mim/
        └── train.txt                    # Diplomatic (resampled) + Normalized + MIM-GOLD-NER

Usage:
    python create_diplomatic_experiment_splits.py
    
Then for each experiment, use:
    - Train: experiments_diplomatic/<experiment_name>/train.txt
    - Dev:   experiments_diplomatic/shared/dev.txt
    - Test:  experiments_diplomatic/shared/test.txt
"""

import os
import glob
import math
import random
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set
from datetime import datetime

# =============================================================================
# CONFIGURATION - EDIT PATHS HERE
# =============================================================================

# Diplomatic Old Icelandic ConLL files (TARGET DOMAIN - for training AND evaluation)
# Update these paths to match your diplomatic data location
DIPLOMATIC_FILES = [
    # Example paths - UPDATE THESE to your actual diplomatic file paths
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_alexander_saga.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_am162b0.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_am162bk.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_drauma_jons.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_islendingabok.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_laknisbok.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_olafs_saga.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_voluspa.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_wormianus.txt"
]

# Normalized Old Icelandic ConLL files (AUXILIARY DATA - for training only)
# These are the same normalized files you used before
NORMALIZED_FILES = [
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1210_jartein_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1210_thorlakur_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1250_sturlunga_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1250_thetubrot_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1260_jomsvikingar_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1275_morkin_conll.txt"
]

# MIM-GOLD-NER directory (Modern Icelandic - SOURCE DOMAIN for cross-domain training)
# This tests whether modern Icelandic helps diplomatic Old Icelandic NER
MIM_GOLD_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/mim_gold_ner/filtered"

# Output directory for diplomatic experiments
OUTPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_diplomatic"

# =============================================================================
# SPLIT CONFIGURATION
# =============================================================================

# Target ratios for dev and test (of total diplomatic sentences)
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# What fraction of LOCATION-containing sentences go to evaluation (dev+test)?
# Using same ratio as normalized experiments for consistency
LOCATION_EVAL_RATIO = 0.5

# Resampling method for experiments that use resampling
RESAMPLING_METHOD = 'sCR'

# Random seed - CRITICAL: Keep this constant for reproducibility!
# Using DIFFERENT seed than normalized experiments to ensure independence
# (Or use same seed if you want parallel sentence selection)
RANDOM_SEED = 42

# =============================================================================
# CONLL PARSING
# =============================================================================

def parse_conll_file(filepath: str) -> List[List[Tuple[str, str]]]:
    """Parse a CoNLL file into sentences."""
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


def write_conll_file(filepath: str, sentences: List[List[Tuple[str, str]]]):
    """Write sentences to a CoNLL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            for token, label in sentence:
                f.write(f"{token}\t{label}\n")
            if i < len(sentences) - 1:
                f.write('\n')


# =============================================================================
# ENTITY ANALYSIS
# =============================================================================

def get_entity_types(sentence: List[Tuple[str, str]]) -> Set[str]:
    """Get unique entity types in a sentence."""
    types = set()
    for token, label in sentence:
        if label.startswith('B-') or label.startswith('I-'):
            types.add(label[2:])
    return types


def count_entities(sentence: List[Tuple[str, str]]) -> Dict[str, int]:
    """Count B- tags (entity starts) by type."""
    counts = defaultdict(int)
    for token, label in sentence:
        if label.startswith('B-'):
            counts[label[2:]] += 1
    return dict(counts)


def count_entity_tokens(sentence: List[Tuple[str, str]]) -> Dict[str, int]:
    """Count all entity tokens (B- and I-) by type."""
    counts = defaultdict(int)
    for token, label in sentence:
        if label.startswith('B-') or label.startswith('I-'):
            counts[label[2:]] += 1
    return dict(counts)


def get_corpus_stats(sentences: List[List[Tuple[str, str]]]) -> Dict:
    """Get comprehensive corpus statistics."""
    total_tokens = 0
    entity_counts = defaultdict(int)
    token_counts = defaultdict(int)
    sentences_with_entity = defaultdict(int)
    
    for sentence in sentences:
        total_tokens += len(sentence)
        seen_types = set()
        for token, label in sentence:
            if label.startswith('B-'):
                etype = label[2:]
                entity_counts[etype] += 1
                seen_types.add(etype)
            if label.startswith('B-') or label.startswith('I-'):
                etype = label[2:]
                token_counts[etype] += 1
        for etype in seen_types:
            sentences_with_entity[etype] += 1
    
    # Compute rareness (for resampling)
    rareness = {}
    for etype, count in token_counts.items():
        prob = count / total_tokens if total_tokens > 0 else 0
        rareness[etype] = -math.log2(prob) if prob > 0 else 0
    
    return {
        'total_sentences': len(sentences),
        'total_tokens': total_tokens,
        'entity_counts': dict(entity_counts),
        'token_counts': dict(token_counts),
        'sentences_with_entity': dict(sentences_with_entity),
        'rareness': rareness,
    }


# =============================================================================
# STRATIFIED SPLITTING
# =============================================================================

def split_list_three_way(
    items: List,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List, List, List]:
    """Split a list into train/dev/test with given ratios."""
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_dev = int(n * dev_ratio / (train_ratio + dev_ratio + test_ratio))
    n_test = int(n * test_ratio / (train_ratio + dev_ratio + test_ratio))
    n_train = n - n_dev - n_test
    
    return shuffled[:n_train], shuffled[n_train:n_train+n_dev], shuffled[n_train+n_dev:]


def stratified_split(
    sentences: List[List[Tuple[str, str]]],
    dev_ratio: float,
    test_ratio: float,
    location_eval_ratio: float,
    seed: int
) -> Tuple[List, List, List]:
    """
    Stratified split prioritizing Location entities in dev/test.
    """
    random.seed(seed)
    train_ratio = 1.0 - dev_ratio - test_ratio
    
    # Categorize sentences
    location_only = []
    person_only = []
    both = []
    neither = []
    
    for sentence in sentences:
        types = get_entity_types(sentence)
        has_loc = 'Location' in types
        has_per = 'Person' in types
        
        if has_loc and has_per:
            both.append(sentence)
        elif has_loc:
            location_only.append(sentence)
        elif has_per:
            person_only.append(sentence)
        else:
            neither.append(sentence)
    
    print(f"\n   Sentence categorization:")
    print(f"     Location-only: {len(location_only)} sentences")
    print(f"     Person-only:   {len(person_only)} sentences")
    print(f"     Both:          {len(both)} sentences")
    print(f"     Neither:       {len(neither)} sentences")
    
    # Location-containing sentences: special ratio for evaluation boost
    location_sentences = location_only + both
    random.seed(seed)
    random.shuffle(location_sentences)
    
    n_loc = len(location_sentences)
    n_loc_eval = int(n_loc * location_eval_ratio)
    n_loc_train = n_loc - n_loc_eval
    n_loc_dev = n_loc_eval // 2
    n_loc_test = n_loc_eval - n_loc_dev
    
    loc_train = location_sentences[:n_loc_train]
    loc_dev = location_sentences[n_loc_train:n_loc_train + n_loc_dev]
    loc_test = location_sentences[n_loc_train + n_loc_dev:]
    
    print(f"\n   Location-containing sentences allocation:")
    print(f"     Total: {n_loc}")
    print(f"     → Train: {len(loc_train)} ({100*len(loc_train)/n_loc:.1f}%)")
    print(f"     → Dev:   {len(loc_dev)} ({100*len(loc_dev)/n_loc:.1f}%)")
    print(f"     → Test:  {len(loc_test)} ({100*len(loc_test)/n_loc:.1f}%)")
    
    # Person-only: standard ratios
    per_train, per_dev, per_test = split_list_three_way(
        person_only, train_ratio, dev_ratio, test_ratio, seed
    )
    
    print(f"\n   Person-only sentences allocation:")
    print(f"     Total: {len(person_only)}")
    print(f"     → Train: {len(per_train)}")
    print(f"     → Dev:   {len(per_dev)}")
    print(f"     → Test:  {len(per_test)}")
    
    # No-entity: standard ratios
    nei_train, nei_dev, nei_test = split_list_three_way(
        neither, train_ratio, dev_ratio, test_ratio, seed
    )
    
    print(f"\n   No-entity sentences allocation:")
    print(f"     Total: {len(neither)}")
    print(f"     → Train: {len(nei_train)}")
    print(f"     → Dev:   {len(nei_dev)}")
    print(f"     → Test:  {len(nei_test)}")
    
    # Combine
    train = loc_train + per_train + nei_train
    dev = loc_dev + per_dev + nei_dev
    test = loc_test + per_test + nei_test
    
    # Shuffle each split
    for lst in [train, dev, test]:
        random.shuffle(lst)
    
    return train, dev, test


# =============================================================================
# SENTENCE-LEVEL RESAMPLING
# =============================================================================

def compute_resampling_factor(
    sentence: List[Tuple[str, str]],
    stats: Dict,
    method: str
) -> int:
    """Compute resampling factor for a sentence."""
    entity_tokens = count_entity_tokens(sentence)
    rareness = stats['rareness']
    length = len(sentence)
    
    if not entity_tokens:
        return 1
    
    if method == 'sC':
        # Equation 1: Just count entity tokens
        return 1 + sum(entity_tokens.values())
    
    elif method == 'sCR':
        # Equation 2: Count weighted by rareness, with sqrt dampening
        weighted = sum(rareness.get(t, 1) * c for t, c in entity_tokens.items())
        return 1 + math.ceil(math.sqrt(weighted)) if weighted > 0 else 1
    
    elif method == 'sCRD':
        # Equation 3: Add density factor (divide by sqrt of sentence length)
        weighted = sum(rareness.get(t, 1) * c for t, c in entity_tokens.items())
        return 1 + math.ceil(weighted / math.sqrt(length)) if weighted > 0 and length > 0 else 1
    
    elif method == 'nsCRD':
        # Equation 4: Add diminishing marginal utility (sqrt of count)
        weighted = sum(rareness.get(t, 1) * math.sqrt(c) for t, c in entity_tokens.items())
        return 1 + math.ceil(weighted / math.sqrt(length)) if weighted > 0 and length > 0 else 1
    
    return 1


def resample_sentences(
    sentences: List[List[Tuple[str, str]]],
    method: str,
    seed: int
) -> Tuple[List, Dict]:
    """Resample sentences based on entity composition."""
    stats = get_corpus_stats(sentences)
    
    factors = []
    resampled = []
    
    for sentence in sentences:
        factor = compute_resampling_factor(sentence, stats, method)
        factors.append(factor)
        for _ in range(factor):
            resampled.append(sentence)
    
    random.seed(seed)
    random.shuffle(resampled)
    
    return resampled, {
        'original': len(sentences),
        'resampled': len(resampled),
        'expansion': len(resampled) / len(sentences) if sentences else 0,
        'factors': Counter(factors),
        'rareness': stats['rareness'],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Diplomatic Old Icelandic NER Experiment Configuration Generator")
    print("=" * 70)
    print(f"\nCreating consistent experimental setup for DIPLOMATIC texts...")
    print(f"Random seed: {RANDOM_SEED} (fixed for reproducibility)")
    print(f"Output directory: {OUTPUT_DIR}")
    
    random.seed(RANDOM_SEED)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "shared"), exist_ok=True)
    
    # =========================================================================
    # 1. Load Diplomatic Old Icelandic (TARGET DOMAIN)
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. Loading Diplomatic Old Icelandic (target domain)")
    print("=" * 70)
    
    diplomatic = []
    files_found = 0
    files_missing = 0
    
    for filepath in DIPLOMATIC_FILES:
        if os.path.exists(filepath):
            sents = parse_conll_file(filepath)
            diplomatic.extend(sents)
            print(f"   ✓ {os.path.basename(filepath)}: {len(sents)} sentences")
            files_found += 1
        else:
            print(f"   ✗ {os.path.basename(filepath)}: NOT FOUND")
            files_missing += 1
    
    if files_missing > 0:
        print(f"\n   WARNING: {files_missing} files not found!")
        print(f"   Please update DIPLOMATIC_FILES paths in the script.")
    
    if len(diplomatic) == 0:
        print("\n   ERROR: No diplomatic data loaded!")
        print("   Please check your file paths and try again.")
        return
    
    print(f"\n   Total diplomatic sentences: {len(diplomatic)}")
    
    diplo_stats = get_corpus_stats(diplomatic)
    print(f"\n   Entity distribution:")
    total_entities = sum(diplo_stats['entity_counts'].values())
    for etype, count in sorted(diplo_stats['entity_counts'].items()):
        pct = 100 * count / total_entities if total_entities > 0 else 0
        sents = diplo_stats['sentences_with_entity'].get(etype, 0)
        print(f"     {etype}: {count} entities ({pct:.1f}%), in {sents} sentences")
    
    # =========================================================================
    # 2. Load Normalized Old Icelandic (AUXILIARY DATA)
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. Loading Normalized Old Icelandic (auxiliary training data)")
    print("=" * 70)
    
    normalized = []
    for filepath in NORMALIZED_FILES:
        if os.path.exists(filepath):
            sents = parse_conll_file(filepath)
            normalized.extend(sents)
            print(f"   ✓ {os.path.basename(filepath)}: {len(sents)} sentences")
        else:
            print(f"   ✗ {os.path.basename(filepath)}: NOT FOUND")
    
    print(f"\n   Total normalized sentences: {len(normalized)}")
    
    if len(normalized) > 0:
        norm_stats = get_corpus_stats(normalized)
        print(f"   Entities: ", end="")
        for etype, count in sorted(norm_stats['entity_counts'].items()):
            print(f"{etype}={count:,} ", end="")
        print()
    
    # =========================================================================
    # 2b. Load MIM-GOLD-NER (Modern Icelandic - SOURCE DOMAIN)
    # =========================================================================
    print("\n" + "=" * 70)
    print("2b. Loading MIM-GOLD-NER (Modern Icelandic - source domain)")
    print("=" * 70)
    
    mim_sentences = []
    if os.path.exists(MIM_GOLD_DIR):
        for filepath in glob.glob(os.path.join(MIM_GOLD_DIR, "*.txt")):
            sents = parse_conll_file(filepath)
            mim_sentences.extend(sents)
        print(f"   ✓ Loaded from: {MIM_GOLD_DIR}")
        print(f"   Total MIM-GOLD-NER sentences: {len(mim_sentences):,}")
        
        if len(mim_sentences) > 0:
            mim_stats = get_corpus_stats(mim_sentences)
            print(f"   Entities: ", end="")
            for etype, count in sorted(mim_stats['entity_counts'].items()):
                print(f"{etype}={count:,} ", end="")
            print()
    else:
        print(f"   ✗ Directory not found: {MIM_GOLD_DIR}")
        print(f"   MIM-GOLD-NER experiments will be skipped.")
    
    # =========================================================================
    # 3. Create SHARED dev/test from DIPLOMATIC texts only
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. Creating SHARED dev/test splits (DIPLOMATIC texts only)")
    print("=" * 70)
    print("   These will be used for ALL diplomatic experiments!")
    
    diplo_train, diplo_dev, diplo_test = stratified_split(
        diplomatic,
        DEV_RATIO,
        TEST_RATIO,
        LOCATION_EVAL_RATIO,
        RANDOM_SEED
    )
    
    # Write shared dev/test
    dev_path = os.path.join(OUTPUT_DIR, "shared", "dev.txt")
    test_path = os.path.join(OUTPUT_DIR, "shared", "test.txt")
    
    write_conll_file(dev_path, diplo_dev)
    write_conll_file(test_path, diplo_test)
    
    dev_stats = get_corpus_stats(diplo_dev)
    test_stats = get_corpus_stats(diplo_test)
    train_stats = get_corpus_stats(diplo_train)
    
    print(f"\n   Split results:")
    
    train_loc_pct = 100 * train_stats['entity_counts'].get('Location', 0) / sum(train_stats['entity_counts'].values()) if sum(train_stats['entity_counts'].values()) > 0 else 0
    print(f"     Train: {len(diplo_train):,} sentences | Person={train_stats['entity_counts'].get('Person', 0):,}, Location={train_stats['entity_counts'].get('Location', 0):,} ({train_loc_pct:.1f}%)")
    
    dev_loc_pct = 100 * dev_stats['entity_counts'].get('Location', 0) / sum(dev_stats['entity_counts'].values()) if sum(dev_stats['entity_counts'].values()) > 0 else 0
    print(f"     Dev:   {len(diplo_dev):,} sentences | Person={dev_stats['entity_counts'].get('Person', 0):,}, Location={dev_stats['entity_counts'].get('Location', 0):,} ({dev_loc_pct:.1f}%)")
    
    test_loc_pct = 100 * test_stats['entity_counts'].get('Location', 0) / sum(test_stats['entity_counts'].values()) if sum(test_stats['entity_counts'].values()) > 0 else 0
    print(f"     Test:  {len(diplo_test):,} sentences | Person={test_stats['entity_counts'].get('Person', 0):,}, Location={test_stats['entity_counts'].get('Location', 0):,} ({test_loc_pct:.1f}%)")
    
    print(f"\n   SHARED DEV SET written to: {dev_path}")
    print(f"   SHARED TEST SET written to: {test_path}")
    
    # =========================================================================
    # 4. Create resampled Diplomatic training
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"4. Resampling Diplomatic training ({RESAMPLING_METHOD})")
    print("=" * 70)
    
    diplo_train_resampled, resamp_stats = resample_sentences(
        diplo_train, RESAMPLING_METHOD, RANDOM_SEED
    )
    
    print(f"\n   Rareness scores (from diplomatic train):")
    for etype, rare in sorted(resamp_stats['rareness'].items(), key=lambda x: -x[1]):
        print(f"     {etype}: {rare:.2f}")
    
    print(f"\n   Resampling results:")
    print(f"     Original: {resamp_stats['original']:,} sentences")
    print(f"     Resampled: {resamp_stats['resampled']:,} sentences")
    print(f"     Expansion: {resamp_stats['expansion']:.2f}x")
    
    print(f"\n   Factor distribution:")
    for factor, count in sorted(resamp_stats['factors'].items()):
        pct = 100 * count / resamp_stats['original']
        bar = '█' * int(pct / 2)
        print(f"     {factor}x: {count:5} ({pct:5.1f}%) {bar}")
    
    # =========================================================================
    # 5. Create experiment-specific training sets
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. Creating experiment-specific training sets")
    print("=" * 70)
    
    experiments = {}
    
    # Experiment 1: diplo_only (Diplomatic only, no resampling)
    exp_name = "diplo_only"
    os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
    train_data = diplo_train.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(train_data)
    train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
    write_conll_file(train_path, train_data)
    exp_stats = get_corpus_stats(train_data)
    experiments[exp_name] = {
        'description': 'Diplomatic only (no resampling)',
        'train_sentences': len(train_data),
        'diplo_sentences': len(diplo_train),
        'norm_sentences': 0,
        'resampled': False,
        'entities': exp_stats['entity_counts'],
        'train_path': train_path,
    }
    print(f"\n   {exp_name}:")
    print(f"     {experiments[exp_name]['description']}")
    print(f"     Train: {len(train_data):,} sentences")
    print(f"     Written to: {train_path}")
    
    # Experiment 2: diplo_resamp (Diplomatic only, resampled)
    exp_name = "diplo_resamp"
    os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
    train_data = diplo_train_resampled.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(train_data)
    train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
    write_conll_file(train_path, train_data)
    exp_stats = get_corpus_stats(train_data)
    experiments[exp_name] = {
        'description': 'Diplomatic only (resampled)',
        'train_sentences': len(train_data),
        'diplo_sentences': len(diplo_train_resampled),
        'diplo_original': len(diplo_train),
        'norm_sentences': 0,
        'resampled': True,
        'resampling_method': RESAMPLING_METHOD,
        'expansion_factor': resamp_stats['expansion'],
        'entities': exp_stats['entity_counts'],
        'train_path': train_path,
    }
    print(f"\n   {exp_name}:")
    print(f"     {experiments[exp_name]['description']}")
    print(f"     Train: {len(train_data):,} sentences ({resamp_stats['expansion']:.2f}x expansion)")
    print(f"     Written to: {train_path}")
    
    # Experiment 3: diplo_plus_norm (Diplomatic + Normalized, no resampling)
    if len(normalized) > 0:
        exp_name = "diplo_plus_norm"
        os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
        train_data = diplo_train + normalized
        random.seed(RANDOM_SEED)
        random.shuffle(train_data)
        train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
        write_conll_file(train_path, train_data)
        exp_stats = get_corpus_stats(train_data)
        experiments[exp_name] = {
            'description': 'Diplomatic + Normalized Old Icelandic',
            'train_sentences': len(train_data),
            'diplo_sentences': len(diplo_train),
            'norm_sentences': len(normalized),
            'resampled': False,
            'entities': exp_stats['entity_counts'],
            'train_path': train_path,
        }
        print(f"\n   {exp_name}:")
        print(f"     {experiments[exp_name]['description']}")
        print(f"     Train: {len(train_data):,} sentences (diplo: {len(diplo_train):,} + norm: {len(normalized):,})")
        print(f"     Written to: {train_path}")
        
        # Experiment 4: diplo_resamp_plus_norm (Diplomatic resampled + Normalized)
        exp_name = "diplo_resamp_plus_norm"
        os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
        train_data = diplo_train_resampled + normalized
        random.seed(RANDOM_SEED)
        random.shuffle(train_data)
        train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
        write_conll_file(train_path, train_data)
        exp_stats = get_corpus_stats(train_data)
        experiments[exp_name] = {
            'description': 'Diplomatic (resampled) + Normalized Old Icelandic',
            'train_sentences': len(train_data),
            'diplo_sentences': len(diplo_train_resampled),
            'diplo_original': len(diplo_train),
            'norm_sentences': len(normalized),
            'resampled': True,
            'resampling_method': RESAMPLING_METHOD,
            'expansion_factor': resamp_stats['expansion'],
            'entities': exp_stats['entity_counts'],
            'train_path': train_path,
        }
        print(f"\n   {exp_name}:")
        print(f"     {experiments[exp_name]['description']}")
        print(f"     Train: {len(train_data):,} sentences (diplo resampled: {len(diplo_train_resampled):,} + norm: {len(normalized):,})")
        print(f"     Written to: {train_path}")
    else:
        print("\n   NOTE: Normalized data not found, skipping combined experiments.")
    
    # =========================================================================
    # MIM-GOLD-NER EXPERIMENTS (Modern Icelandic)
    # =========================================================================
    if len(mim_sentences) > 0:
        print("\n" + "-" * 70)
        print("   MIM-GOLD-NER experiments (Modern Icelandic)")
        print("-" * 70)
        
        # Experiment: diplo_plus_mim (Diplomatic + MIM-GOLD-NER)
        exp_name = "diplo_plus_mim"
        os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
        train_data = diplo_train + mim_sentences
        random.seed(RANDOM_SEED)
        random.shuffle(train_data)
        train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
        write_conll_file(train_path, train_data)
        exp_stats = get_corpus_stats(train_data)
        experiments[exp_name] = {
            'description': 'Diplomatic + MIM-GOLD-NER (Modern Icelandic)',
            'train_sentences': len(train_data),
            'diplo_sentences': len(diplo_train),
            'mim_sentences': len(mim_sentences),
            'norm_sentences': 0,
            'resampled': False,
            'entities': exp_stats['entity_counts'],
            'train_path': train_path,
        }
        print(f"\n   {exp_name}:")
        print(f"     {experiments[exp_name]['description']}")
        print(f"     Train: {len(train_data):,} sentences (diplo: {len(diplo_train):,} + mim: {len(mim_sentences):,})")
        print(f"     Written to: {train_path}")
        
        # Experiment: diplo_resamp_plus_mim (Diplomatic resampled + MIM-GOLD-NER)
        exp_name = "diplo_resamp_plus_mim"
        os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
        train_data = diplo_train_resampled + mim_sentences
        random.seed(RANDOM_SEED)
        random.shuffle(train_data)
        train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
        write_conll_file(train_path, train_data)
        exp_stats = get_corpus_stats(train_data)
        experiments[exp_name] = {
            'description': 'Diplomatic (resampled) + MIM-GOLD-NER',
            'train_sentences': len(train_data),
            'diplo_sentences': len(diplo_train_resampled),
            'diplo_original': len(diplo_train),
            'mim_sentences': len(mim_sentences),
            'norm_sentences': 0,
            'resampled': True,
            'resampling_method': RESAMPLING_METHOD,
            'expansion_factor': resamp_stats['expansion'],
            'entities': exp_stats['entity_counts'],
            'train_path': train_path,
        }
        print(f"\n   {exp_name}:")
        print(f"     {experiments[exp_name]['description']}")
        print(f"     Train: {len(train_data):,} sentences (diplo resampled: {len(diplo_train_resampled):,} + mim: {len(mim_sentences):,})")
        print(f"     Written to: {train_path}")
        
        # Combined experiments with ALL data sources
        if len(normalized) > 0:
            print("\n" + "-" * 70)
            print("   Combined experiments (Diplomatic + Normalized + MIM-GOLD-NER)")
            print("-" * 70)
            
            # Experiment: diplo_plus_norm_plus_mim (All three, no resampling)
            exp_name = "diplo_plus_norm_plus_mim"
            os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
            train_data = diplo_train + normalized + mim_sentences
            random.seed(RANDOM_SEED)
            random.shuffle(train_data)
            train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
            write_conll_file(train_path, train_data)
            exp_stats = get_corpus_stats(train_data)
            experiments[exp_name] = {
                'description': 'Diplomatic + Normalized + MIM-GOLD-NER',
                'train_sentences': len(train_data),
                'diplo_sentences': len(diplo_train),
                'norm_sentences': len(normalized),
                'mim_sentences': len(mim_sentences),
                'resampled': False,
                'entities': exp_stats['entity_counts'],
                'train_path': train_path,
            }
            print(f"\n   {exp_name}:")
            print(f"     {experiments[exp_name]['description']}")
            print(f"     Train: {len(train_data):,} sentences (diplo: {len(diplo_train):,} + norm: {len(normalized):,} + mim: {len(mim_sentences):,})")
            print(f"     Written to: {train_path}")
            
            # Experiment: diplo_resamp_plus_norm_plus_mim (All three, diplo resampled)
            exp_name = "diplo_resamp_plus_norm_plus_mim"
            os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
            train_data = diplo_train_resampled + normalized + mim_sentences
            random.seed(RANDOM_SEED)
            random.shuffle(train_data)
            train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
            write_conll_file(train_path, train_data)
            exp_stats = get_corpus_stats(train_data)
            experiments[exp_name] = {
                'description': 'Diplomatic (resampled) + Normalized + MIM-GOLD-NER',
                'train_sentences': len(train_data),
                'diplo_sentences': len(diplo_train_resampled),
                'diplo_original': len(diplo_train),
                'norm_sentences': len(normalized),
                'mim_sentences': len(mim_sentences),
                'resampled': True,
                'resampling_method': RESAMPLING_METHOD,
                'expansion_factor': resamp_stats['expansion'],
                'entities': exp_stats['entity_counts'],
                'train_path': train_path,
            }
            print(f"\n   {exp_name}:")
            print(f"     {experiments[exp_name]['description']}")
            print(f"     Train: {len(train_data):,} sentences (diplo resampled: {len(diplo_train_resampled):,} + norm: {len(normalized):,} + mim: {len(mim_sentences):,})")
            print(f"     Written to: {train_path}")
    else:
        print("\n   NOTE: MIM-GOLD-NER not found, skipping Modern Icelandic experiments.")
    
    # =========================================================================
    # 6. Save experiment configuration
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. Saving experiment configuration")
    print("=" * 70)
    
    config = {
        'created': datetime.now().isoformat(),
        'experiment_type': 'diplomatic',
        'random_seed': RANDOM_SEED,
        'shared': {
            'dev_path': dev_path,
            'test_path': test_path,
            'dev_sentences': len(diplo_dev),
            'test_sentences': len(diplo_test),
            'dev_entities': dev_stats['entity_counts'],
            'test_entities': test_stats['entity_counts'],
            'dev_location_pct': dev_loc_pct,
            'test_location_pct': test_loc_pct,
        },
        'experiments': experiments,
        'source_files': {
            'diplomatic': DIPLOMATIC_FILES,
            'normalized': NORMALIZED_FILES,
            'mim_gold_dir': MIM_GOLD_DIR,
        },
        'split_config': {
            'dev_ratio': DEV_RATIO,
            'test_ratio': TEST_RATIO,
            'location_eval_ratio': LOCATION_EVAL_RATIO,
            'resampling_method': RESAMPLING_METHOD,
        }
    }
    
    config_path = os.path.join(OUTPUT_DIR, "experiment_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"   Configuration saved to: {config_path}")
    
    # =========================================================================
    # 7. Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIPLOMATIC EXPERIMENT SETUP COMPLETE")
    print("=" * 70)
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ SHARED EVALUATION SETS (DIPLOMATIC only - same for ALL experiments) │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Dev:  {len(diplo_dev):,} sentences | Person={dev_stats['entity_counts'].get('Person', 0):,}, Location={dev_stats['entity_counts'].get('Location', 0):,} ({dev_loc_pct:.1f}%)     │
  │ Test: {len(diplo_test):,} sentences | Person={test_stats['entity_counts'].get('Person', 0):,}, Location={test_stats['entity_counts'].get('Location', 0):,} ({test_loc_pct:.1f}%)    │
  └─────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ EXPERIMENT TRAINING SETS                                            │
  ├─────────────────────────────────────────────────────────────────────┤""")
    
    for exp_name, exp_data in experiments.items():
        print(f"  │ {exp_name:<25} {exp_data['train_sentences']:>7,} sentences              │")
        print(f"  │   └── {exp_data['description']:<55} │")
    
    print(f"""  └─────────────────────────────────────────────────────────────────────┘

  DIRECTORY STRUCTURE:
  {OUTPUT_DIR}/
  ├── shared/
  │   ├── dev.txt                      # Use for ALL diplomatic experiments
  │   └── test.txt                     # Use for ALL diplomatic experiments
  │
  │── BASE EXPERIMENTS (no resampling):
  ├── diplo_only/                      # Diplomatic only
  ├── diplo_plus_norm/                 # Diplomatic + Normalized Old Icelandic
  ├── diplo_plus_mim/                  # Diplomatic + MIM-GOLD-NER (Modern Icelandic)
  ├── diplo_plus_norm_plus_mim/        # Diplomatic + Normalized + MIM-GOLD-NER
  │
  │── RESAMPLED EXPERIMENTS:
  ├── diplo_resamp/                    # Diplomatic (resampled)
  ├── diplo_resamp_plus_norm/          # Diplomatic (resampled) + Normalized
  ├── diplo_resamp_plus_mim/           # Diplomatic (resampled) + MIM-GOLD-NER
  ├── diplo_resamp_plus_norm_plus_mim/ # Diplomatic (resampled) + Norm + MIM
  │
  └── experiment_config.json

  RESEARCH QUESTIONS ANSWERED:
  ┌───────────────────────────────────────────────────────────────────────────┐
  │ Q1: Does fine-tuning IceBERT help?                                        │
  │     Compare: Original IceBERT vs Fine-tuned IceBERT (same training data)  │
  ├───────────────────────────────────────────────────────────────────────────┤
  │ Q2: Does Normalized Old Icelandic help diplomatic NER?                    │
  │     Compare: diplo_plus_norm vs diplo_only                                │
  ├───────────────────────────────────────────────────────────────────────────┤
  │ Q3: Does Modern Icelandic (MIM-GOLD-NER) help diplomatic NER?             │
  │     Compare: diplo_plus_mim vs diplo_only                                 │
  ├───────────────────────────────────────────────────────────────────────────┤
  │ Q4: Does auxiliary data need to be historical?                            │
  │     Compare: diplo_plus_norm vs diplo_plus_mim                            │
  │     (Same domain OLD vs different domain MODERN)                          │
  ├───────────────────────────────────────────────────────────────────────────┤
  │ Q5: Does combining ALL data sources help most?                            │
  │     Compare: diplo_plus_norm_plus_mim vs individual additions             │
  ├───────────────────────────────────────────────────────────────────────────┤
  │ Q6: How does resampling interact with data augmentation?                  │
  │     Compare: resampled versions vs non-resampled versions                 │
  └───────────────────────────────────────────────────────────────────────────┘

  SUGGESTED EXPERIMENT ORDER:
  ┌────┬─────────────────────┬──────────────────────────────────────────────────┐
  │ #  │ Model               │ Training Data                                    │
  ├────┼─────────────────────┼──────────────────────────────────────────────────┤
  │ 1  │ Original IceBERT    │ diplo_only                                       │
  │ 2  │ Fine-tuned IceBERT  │ diplo_only                                       │
  ├────┼─────────────────────┼──────────────────────────────────────────────────┤
  │    │ → Pick best model from 1-2, use for remaining experiments             │
  ├────┼─────────────────────┼──────────────────────────────────────────────────┤
  │ 3  │ Best model          │ diplo_plus_norm (+ Normalized Old Icelandic)     │
  │ 4  │ Best model          │ diplo_plus_mim (+ Modern Icelandic)              │
  │ 5  │ Best model          │ diplo_plus_norm_plus_mim (+ Both)                │
  ├────┼─────────────────────┼──────────────────────────────────────────────────┤
  │ 6  │ Best model          │ diplo_resamp (resampled diplomatic)              │
  │ 7  │ Best model          │ diplo_resamp_plus_norm                           │
  │ 8  │ Best model          │ diplo_resamp_plus_mim                            │
  │ 9  │ Best model          │ diplo_resamp_plus_norm_plus_mim                  │
  └────┴─────────────────────┴──────────────────────────────────────────────────┘

  USAGE EXAMPLE (for each experiment):
    TRAIN_FILE = "experiments_diplomatic/<experiment>/train.txt"
    DEV_FILE   = "experiments_diplomatic/shared/dev.txt"     # ALWAYS the same
    TEST_FILE  = "experiments_diplomatic/shared/test.txt"    # ALWAYS the same
""")
    
    print("=" * 70)
    print("Done! Your diplomatic experimental setup is ready.")
    print("=" * 70)
    print("\nIMPORTANT: Update the file paths in DIPLOMATIC_FILES at the top")
    print("of this script to match your actual diplomatic data location!")


if __name__ == "__main__":
    main()