#!/usr/bin/env python3
"""
NER Experiment Configuration Generator

This script creates a consistent experimental setup for comparing different
NER training approaches while keeping dev/test sets IDENTICAL across all experiments.

Experiments:
1. baseline_mim:        Original IceBERT + MIM-GOLD-NER + Old Icelandic (no resampling)
2. baseline_mim_resamp: Original IceBERT + MIM-GOLD-NER + Old Icelandic (resampled)
3. old_ice_only:        Original IceBERT + Old Icelandic only (no MIM-GOLD-NER, no resampling)
4. old_ice_resamp:      Original IceBERT + Old Icelandic only (resampled, no MIM-GOLD-NER)

For fine-tuned IceBERT experiments, use the same training files but change the model checkpoint.

Key principle: Dev and Test sets are created ONCE and shared across ALL experiments.
Only training data varies.

Directory structure created:
    experiments/
    ├── shared/
    │   ├── dev.txt          # SAME for all experiments
    │   └── test.txt         # SAME for all experiments
    ├── baseline_mim/
    │   └── train.txt        # MIM-GOLD + Old Icelandic (no resampling)
    ├── baseline_mim_resamp/
    │   └── train.txt        # MIM-GOLD + Old Icelandic (resampled)
    ├── old_ice_only/
    │   └── train.txt        # Old Icelandic only (no resampling)
    └── old_ice_resamp/
        └── train.txt        # Old Icelandic only (resampled)

Usage:
    python create_experiment_splits.py
    
Then for each experiment, use:
    - Train: experiments/<experiment_name>/train.txt
    - Dev:   experiments/shared/dev.txt
    - Test:  experiments/shared/test.txt
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

# Old Norse/Old Icelandic ConLL files (TARGET DOMAIN)
OLD_NORSE_FILES = [
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1210_jartein_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1210_thorlakur_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1250_sturlunga_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1250_thetubrot_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1260_jomsvikingar_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/1275_morkin_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/alexander_saga_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/am162b0_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/am162bk_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/drauma_jons_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/islendingabok_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/laknisbok_conll.txt",
    "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/old_icelandic_all/olafs_saga_conll.txt"
]

# MIM-GOLD-NER filtered directory (SOURCE DOMAIN)
MIM_GOLD_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/mim_gold_ner/filtered"

# Output directory for experiments
OUTPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_normalised"

# =============================================================================
# SPLIT CONFIGURATION
# =============================================================================

# Target ratios for dev and test (of total Old Icelandic sentences)
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# What fraction of LOCATION-containing sentences go to evaluation (dev+test)?
LOCATION_EVAL_RATIO = 0.5

# Resampling method for experiments that use resampling
RESAMPLING_METHOD = 'sCR'

# Random seed - CRITICAL: Keep this constant for reproducibility!
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
    
    # Person-only: standard ratios
    per_train, per_dev, per_test = split_list_three_way(
        person_only, train_ratio, dev_ratio, test_ratio, seed
    )
    
    # No-entity: standard ratios
    nei_train, nei_dev, nei_test = split_list_three_way(
        neither, train_ratio, dev_ratio, test_ratio, seed
    )
    
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
        return 1 + sum(entity_tokens.values())
    
    elif method == 'sCR':
        weighted = sum(rareness.get(t, 1) * c for t, c in entity_tokens.items())
        return 1 + math.ceil(math.sqrt(weighted)) if weighted > 0 else 1
    
    elif method == 'sCRD':
        weighted = sum(rareness.get(t, 1) * c for t, c in entity_tokens.items())
        return 1 + math.ceil(weighted / math.sqrt(length)) if weighted > 0 and length > 0 else 1
    
    elif method == 'nsCRD':
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
    print("NER Experiment Configuration Generator")
    print("=" * 70)
    print(f"\nCreating consistent experimental setup...")
    print(f"Random seed: {RANDOM_SEED} (fixed for reproducibility)")
    print(f"Output directory: {OUTPUT_DIR}")
    
    random.seed(RANDOM_SEED)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "shared"), exist_ok=True)
    
    # =========================================================================
    # 1. Load Old Icelandic
    # =========================================================================
    print("\n" + "=" * 70)
    print("1. Loading Old Icelandic (target domain)")
    print("=" * 70)
    
    old_norse = []
    for filepath in OLD_NORSE_FILES:
        if os.path.exists(filepath):
            sents = parse_conll_file(filepath)
            old_norse.extend(sents)
            print(f"   {os.path.basename(filepath)}: {len(sents)} sentences")
        else:
            print(f"   WARNING: {filepath} not found")
    
    print(f"\n   Total: {len(old_norse)} sentences")
    
    on_stats = get_corpus_stats(old_norse)
    print(f"\n   Entity distribution:")
    total_entities = sum(on_stats['entity_counts'].values())
    for etype, count in sorted(on_stats['entity_counts'].items()):
        pct = 100 * count / total_entities if total_entities > 0 else 0
        sents = on_stats['sentences_with_entity'].get(etype, 0)
        print(f"     {etype}: {count} entities ({pct:.1f}%), in {sents} sentences")
    
    # =========================================================================
    # 2. Load MIM-GOLD-NER
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. Loading MIM-GOLD-NER (source domain)")
    print("=" * 70)
    
    mim_sentences = []
    for filepath in glob.glob(os.path.join(MIM_GOLD_DIR, "*.txt")):
        mim_sentences.extend(parse_conll_file(filepath))
    
    print(f"   Total: {len(mim_sentences):,} sentences")
    
    mim_stats = get_corpus_stats(mim_sentences)
    print(f"   Entities: ", end="")
    for etype, count in sorted(mim_stats['entity_counts'].items()):
        print(f"{etype}={count:,} ", end="")
    print()
    
    # =========================================================================
    # 3. Create SHARED dev/test (ONCE - used by ALL experiments)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. Creating SHARED dev/test splits (used by ALL experiments)")
    print("=" * 70)
    
    on_train, on_dev, on_test = stratified_split(
        old_norse,
        DEV_RATIO,
        TEST_RATIO,
        LOCATION_EVAL_RATIO,
        RANDOM_SEED
    )
    
    # Write shared dev/test
    dev_path = os.path.join(OUTPUT_DIR, "shared", "dev.txt")
    test_path = os.path.join(OUTPUT_DIR, "shared", "test.txt")
    
    write_conll_file(dev_path, on_dev)
    write_conll_file(test_path, on_test)
    
    dev_stats = get_corpus_stats(on_dev)
    test_stats = get_corpus_stats(on_test)
    
    print(f"\n   SHARED DEV SET:")
    print(f"     Sentences: {len(on_dev):,}")
    print(f"     Person: {dev_stats['entity_counts'].get('Person', 0):,}")
    print(f"     Location: {dev_stats['entity_counts'].get('Location', 0):,}")
    dev_loc_pct = 100 * dev_stats['entity_counts'].get('Location', 0) / sum(dev_stats['entity_counts'].values())
    print(f"     Location %: {dev_loc_pct:.1f}%")
    print(f"     Written to: {dev_path}")
    
    print(f"\n   SHARED TEST SET:")
    print(f"     Sentences: {len(on_test):,}")
    print(f"     Person: {test_stats['entity_counts'].get('Person', 0):,}")
    print(f"     Location: {test_stats['entity_counts'].get('Location', 0):,}")
    test_loc_pct = 100 * test_stats['entity_counts'].get('Location', 0) / sum(test_stats['entity_counts'].values())
    print(f"     Location %: {test_loc_pct:.1f}%")
    print(f"     Written to: {test_path}")
    
    # =========================================================================
    # 4. Create resampled Old Icelandic training
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"4. Resampling Old Icelandic training ({RESAMPLING_METHOD})")
    print("=" * 70)
    
    on_train_resampled, resamp_stats = resample_sentences(
        on_train, RESAMPLING_METHOD, RANDOM_SEED
    )
    
    print(f"\n   Rareness scores:")
    for etype, rare in sorted(resamp_stats['rareness'].items(), key=lambda x: -x[1]):
        print(f"     {etype}: {rare:.2f}")
    
    print(f"\n   Resampling results:")
    print(f"     Original: {resamp_stats['original']:,} sentences")
    print(f"     Resampled: {resamp_stats['resampled']:,} sentences")
    print(f"     Expansion: {resamp_stats['expansion']:.2f}x")
    
    # =========================================================================
    # 5. Create experiment-specific training sets
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. Creating experiment-specific training sets")
    print("=" * 70)
    
    experiments = {}
    
    # Experiment 1: baseline_mim (MIM-GOLD + Old Icelandic, no resampling)
    exp_name = "baseline_mim"
    os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
    train_data = on_train + mim_sentences
    random.seed(RANDOM_SEED)
    random.shuffle(train_data)
    train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
    write_conll_file(train_path, train_data)
    train_stats = get_corpus_stats(train_data)
    experiments[exp_name] = {
        'description': 'MIM-GOLD-NER + Old Icelandic (no resampling)',
        'train_sentences': len(train_data),
        'old_ice_sentences': len(on_train),
        'mim_sentences': len(mim_sentences),
        'resampled': False,
        'entities': train_stats['entity_counts'],
        'train_path': train_path,
    }
    print(f"\n   {exp_name}:")
    print(f"     {experiments[exp_name]['description']}")
    print(f"     Train: {len(train_data):,} sentences")
    print(f"     Written to: {train_path}")
    
    # Experiment 2: baseline_mim_resamp (MIM-GOLD + Old Icelandic resampled)
    exp_name = "baseline_mim_resamp"
    os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
    train_data = on_train_resampled + mim_sentences
    random.seed(RANDOM_SEED)
    random.shuffle(train_data)
    train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
    write_conll_file(train_path, train_data)
    train_stats = get_corpus_stats(train_data)
    experiments[exp_name] = {
        'description': 'MIM-GOLD-NER + Old Icelandic (resampled)',
        'train_sentences': len(train_data),
        'old_ice_sentences': len(on_train_resampled),
        'old_ice_original': len(on_train),
        'mim_sentences': len(mim_sentences),
        'resampled': True,
        'resampling_method': RESAMPLING_METHOD,
        'expansion_factor': resamp_stats['expansion'],
        'entities': train_stats['entity_counts'],
        'train_path': train_path,
    }
    print(f"\n   {exp_name}:")
    print(f"     {experiments[exp_name]['description']}")
    print(f"     Train: {len(train_data):,} sentences")
    print(f"     Written to: {train_path}")
    
    # Experiment 3: old_ice_only (Old Icelandic only, no resampling)
    exp_name = "old_ice_only"
    os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
    train_data = on_train.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(train_data)
    train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
    write_conll_file(train_path, train_data)
    train_stats = get_corpus_stats(train_data)
    experiments[exp_name] = {
        'description': 'Old Icelandic only (no MIM-GOLD, no resampling)',
        'train_sentences': len(train_data),
        'old_ice_sentences': len(on_train),
        'mim_sentences': 0,
        'resampled': False,
        'entities': train_stats['entity_counts'],
        'train_path': train_path,
    }
    print(f"\n   {exp_name}:")
    print(f"     {experiments[exp_name]['description']}")
    print(f"     Train: {len(train_data):,} sentences")
    print(f"     Written to: {train_path}")
    
    # Experiment 4: old_ice_resamp (Old Icelandic only, resampled)
    exp_name = "old_ice_resamp"
    os.makedirs(os.path.join(OUTPUT_DIR, exp_name), exist_ok=True)
    train_data = on_train_resampled.copy()
    random.seed(RANDOM_SEED)
    random.shuffle(train_data)
    train_path = os.path.join(OUTPUT_DIR, exp_name, "train.txt")
    write_conll_file(train_path, train_data)
    train_stats = get_corpus_stats(train_data)
    experiments[exp_name] = {
        'description': 'Old Icelandic only (resampled, no MIM-GOLD)',
        'train_sentences': len(train_data),
        'old_ice_sentences': len(on_train_resampled),
        'old_ice_original': len(on_train),
        'mim_sentences': 0,
        'resampled': True,
        'resampling_method': RESAMPLING_METHOD,
        'expansion_factor': resamp_stats['expansion'],
        'entities': train_stats['entity_counts'],
        'train_path': train_path,
    }
    print(f"\n   {exp_name}:")
    print(f"     {experiments[exp_name]['description']}")
    print(f"     Train: {len(train_data):,} sentences")
    print(f"     Written to: {train_path}")
    
    # =========================================================================
    # 6. Save experiment configuration
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. Saving experiment configuration")
    print("=" * 70)
    
    config = {
        'created': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'shared': {
            'dev_path': dev_path,
            'test_path': test_path,
            'dev_sentences': len(on_dev),
            'test_sentences': len(on_test),
            'dev_entities': dev_stats['entity_counts'],
            'test_entities': test_stats['entity_counts'],
        },
        'experiments': experiments,
        'source_files': {
            'old_icelandic': OLD_NORSE_FILES,
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
    print("EXPERIMENT SETUP COMPLETE")
    print("=" * 70)
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ SHARED EVALUATION SETS (same for ALL experiments)                   │
  ├─────────────────────────────────────────────────────────────────────┤
  │ Dev:  {len(on_dev):,} sentences | Person={dev_stats['entity_counts'].get('Person', 0):,}, Location={dev_stats['entity_counts'].get('Location', 0):,} ({dev_loc_pct:.1f}%)     │
  │ Test: {len(on_test):,} sentences | Person={test_stats['entity_counts'].get('Person', 0):,}, Location={test_stats['entity_counts'].get('Location', 0):,} ({test_loc_pct:.1f}%)    │
  └─────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ EXPERIMENT TRAINING SETS                                            │
  ├─────────────────────────────────────────────────────────────────────┤""")
    
    for exp_name, exp_data in experiments.items():
        print(f"  │ {exp_name:<20} {exp_data['train_sentences']:>7,} sentences                    │")
        print(f"  │   └── {exp_data['description']:<55} │")
    
    print(f"""  └─────────────────────────────────────────────────────────────────────┘

  DIRECTORY STRUCTURE:
  {OUTPUT_DIR}/
  ├── shared/
  │   ├── dev.txt          # Use for ALL experiments
  │   └── test.txt         # Use for ALL experiments
  ├── baseline_mim/
  │   └── train.txt        # Experiment 1: Original IceBERT + MIM + Old Ice
  ├── baseline_mim_resamp/
  │   └── train.txt        # Experiment 2: Original IceBERT + MIM + Old Ice (resampled)
  ├── old_ice_only/
  │   └── train.txt        # Experiment 3: Original IceBERT + Old Ice only
  ├── old_ice_resamp/
  │   └── train.txt        # Experiment 4: Original IceBERT + Old Ice only (resampled)
  └── experiment_config.json

  FOR FINE-TUNED IceBERT EXPERIMENTS:
  Use the SAME training files, just change the model checkpoint in your training script.
  This gives you 8 total experiments (4 training configs × 2 model versions).

  USAGE EXAMPLE (for each experiment):
    TRAIN_FILE = "experiments/<experiment>/train.txt"
    DEV_FILE   = "experiments/shared/dev.txt"     # ALWAYS the same
    TEST_FILE  = "experiments/shared/test.txt"    # ALWAYS the same
""")
    
    print("=" * 70)
    print("Done! Your experimental setup is ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()