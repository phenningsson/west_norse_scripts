#!/usr/bin/env python3
"""
Domain-Aware Sentence-Level Resampling with Proper Entity Stratification

FIXES from previous version:
1. Training set now gets Location entities (was getting ZERO before)
2. Dev and Test get EQUAL shares of Location-containing sentences
3. Configurable Location priority for evaluation sets

Strategy:
1. Categorize sentences by entity content
2. Split each category independently (not "fill dev first, then test")
3. Allocate more Location to dev/test, but keep some for training
4. Resample only training data

Usage:
    python create_balanced_splits_fixed.py
    python create_balanced_splits_fixed.py --method sCR --location-eval-ratio 0.5
"""

import os
import glob
import math
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set

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

# MIM-GOLD-NER filtered directory (SOURCE DOMAIN - unchanged)
MIM_GOLD_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/mim_gold_ner/filtered"

# Output directory
OUTPUT_DIR = "/Users/phenningsson/Downloads/west_norse_scripts/conll_format_normalised/conll_resampled_splits"

# =============================================================================
# SPLIT CONFIGURATION
# =============================================================================

# Target ratios for dev and test (of total Old Icelandic sentences)
DEV_RATIO = 0.15
TEST_RATIO = 0.15

# What fraction of LOCATION-containing sentences go to evaluation (dev+test)?
# 0.5 = 50% to eval (25% dev, 25% test), 50% to train
# 0.6 = 60% to eval (30% dev, 30% test), 40% to train
# Higher = more reliable Location metrics, but less Location for training
LOCATION_EVAL_RATIO = 0.5

# Resampling method: 'sC', 'sCR', 'sCRD', 'nsCRD', or 'none'
RESAMPLING_METHOD = 'sCR'

# Random seed
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
    entity_counts = defaultdict(int)  # B- tags
    token_counts = defaultdict(int)   # B- and I- tags
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
# PROPER STRATIFIED SPLITTING
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


def stratified_split_proper(
    sentences: List[List[Tuple[str, str]]],
    dev_ratio: float,
    test_ratio: float,
    location_eval_ratio: float,
    seed: int
) -> Tuple[List, List, List]:
    """
    Properly stratified split that:
    1. Keeps some Location entities in training (for learning)
    2. Splits Location sentences equally between dev and test
    3. Allocates more Location to eval for reliable metrics
    """
    random.seed(seed)
    train_ratio = 1.0 - dev_ratio - test_ratio
    
    # Categorize sentences
    location_only = []   # Has Location, no Person
    person_only = []     # Has Person, no Location
    both = []            # Has both Location and Person
    neither = []         # No entities
    
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
    
    # For LOCATION-containing sentences (location_only + both):
    # Use special ratio to get more into eval while keeping some for training
    location_sentences = location_only + both
    random.seed(seed)
    random.shuffle(location_sentences)
    
    # location_eval_ratio goes to dev+test, rest to train
    # Split eval portion equally between dev and test
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
    
    # For PERSON-ONLY sentences: use standard split ratios
    per_train, per_dev, per_test = split_list_three_way(
        person_only, train_ratio, dev_ratio, test_ratio, seed
    )
    
    print(f"\n   Person-only sentences allocation:")
    print(f"     Total: {len(person_only)}")
    print(f"     → Train: {len(per_train)}")
    print(f"     → Dev:   {len(per_dev)}")
    print(f"     → Test:  {len(per_test)}")
    
    # For NO-ENTITY sentences: use standard split ratios
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
    # Compute stats from these sentences
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
    parser = argparse.ArgumentParser(description='Domain-aware NER data splitting with resampling')
    parser.add_argument('--method', default=RESAMPLING_METHOD,
                        choices=['sC', 'sCR', 'sCRD', 'nsCRD', 'none'],
                        help='Resampling method')
    parser.add_argument('--location-eval-ratio', type=float, default=LOCATION_EVAL_RATIO,
                        help='Fraction of Location sentences for dev+test (0.3-0.7)')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    
    use_resampling = args.method != 'none'
    
    print("=" * 70)
    print("Domain-Aware Stratified Splitting with Sentence Resampling")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Location eval ratio: {args.location_eval_ratio*100:.0f}% to dev+test")
    print(f"  Resampling method: {args.method if use_resampling else 'disabled'}")
    print(f"  Dev ratio: {DEV_RATIO*100:.0f}%, Test ratio: {TEST_RATIO*100:.0f}%")
    
    random.seed(args.seed)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
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
    # 2. Stratified split
    # =========================================================================
    print("\n" + "=" * 70)
    print("2. Stratified splitting (proper Location balancing)")
    print("=" * 70)
    
    on_train, on_dev, on_test = stratified_split_proper(
        old_norse,
        DEV_RATIO,
        TEST_RATIO,
        args.location_eval_ratio,
        args.seed
    )
    
    # Show resulting distribution
    print(f"\n   Split results:")
    for name, split in [("Train", on_train), ("Dev", on_dev), ("Test", on_test)]:
        stats = get_corpus_stats(split)
        entities = stats['entity_counts']
        total = sum(entities.values())
        loc_count = entities.get('Location', 0)
        per_count = entities.get('Person', 0)
        loc_pct = 100 * loc_count / total if total > 0 else 0
        print(f"     {name}: {len(split):,} sentences | "
              f"Person={per_count:,}, Location={loc_count:,} ({loc_pct:.1f}%)")
    
    # =========================================================================
    # 3. Resample Old Icelandic training
    # =========================================================================
    if use_resampling:
        print("\n" + "=" * 70)
        print(f"3. Resampling Old Icelandic training ({args.method})")
        print("=" * 70)
        
        on_train_resampled, resamp_stats = resample_sentences(
            on_train, args.method, args.seed
        )
        
        print(f"\n   Rareness scores (from Old Icelandic train):")
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
            print(f"     {factor}x: {count:>5} ({pct:5.1f}%) {bar}")
    else:
        on_train_resampled = on_train
        resamp_stats = {'expansion': 1.0}
        print("\n3. Resampling: DISABLED")
    
    # =========================================================================
    # 4. Load MIM-GOLD-NER (unchanged)
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. Loading MIM-GOLD-NER (source domain - unchanged)")
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
    # 5. Combine and write
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. Creating final splits")
    print("=" * 70)
    
    # Combine training
    train_final = on_train_resampled + mim_sentences
    random.shuffle(train_final)
    
    print(f"\n   Training composition:")
    print(f"     Old Icelandic (resampled): {len(on_train_resampled):,}")
    print(f"     MIM-GOLD-NER (unchanged):  {len(mim_sentences):,}")
    print(f"     Total:                     {len(train_final):,}")
    
    # Write files
    train_path = os.path.join(OUTPUT_DIR, "train.txt")
    dev_path = os.path.join(OUTPUT_DIR, "dev.txt")
    test_path = os.path.join(OUTPUT_DIR, "test.txt")
    
    write_conll_file(train_path, train_final)
    write_conll_file(dev_path, on_dev)
    write_conll_file(test_path, on_test)
    
    print(f"\n   Written:")
    print(f"     {train_path}")
    print(f"     {dev_path}")
    print(f"     {test_path}")
    
    # =========================================================================
    # 6. Final summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    dev_stats = get_corpus_stats(on_dev)
    test_stats = get_corpus_stats(on_test)
    train_on_stats = get_corpus_stats(on_train_resampled)
    
    dev_total = sum(dev_stats['entity_counts'].values())
    test_total = sum(test_stats['entity_counts'].values())
    dev_loc = dev_stats['entity_counts'].get('Location', 0)
    test_loc = test_stats['entity_counts'].get('Location', 0)
    dev_per = dev_stats['entity_counts'].get('Person', 0)
    test_per = test_stats['entity_counts'].get('Person', 0)
    train_loc = train_on_stats['entity_counts'].get('Location', 0)
    train_per = train_on_stats['entity_counts'].get('Person', 0)
    
    dev_loc_pct = 100 * dev_loc / dev_total if dev_total > 0 else 0
    test_loc_pct = 100 * test_loc / test_total if test_total > 0 else 0
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ OLD ICELANDIC TRAINING (before MIM-GOLD-NER added)                  │
  │   Sentences: {len(on_train_resampled):,} (resampled from {len(on_train):,}, {resamp_stats['expansion']:.1f}x)              │
  │   Person:    {train_per:,} entities                                         │
  │   Location:  {train_loc:,} entities ← NOW HAS LOCATION FOR TRAINING!        │
  ├─────────────────────────────────────────────────────────────────────┤
  │ COMBINED TRAINING SET                                               │
  │   Old Icelandic: {len(on_train_resampled):,} + MIM-GOLD-NER: {len(mim_sentences):,} = {len(train_final):,}     │
  ├─────────────────────────────────────────────────────────────────────┤
  │ DEV SET (Old Icelandic only)                                        │
  │   Sentences: {len(on_dev):,}                                                │
  │   Person:    {dev_per:,} entities                                           │
  │   Location:  {dev_loc:,} entities ({dev_loc_pct:.1f}%)                                  │
  ├─────────────────────────────────────────────────────────────────────┤
  │ TEST SET (Old Icelandic only)                                       │
  │   Sentences: {len(on_test):,}                                               │
  │   Person:    {test_per:,} entities                                          │
  │   Location:  {test_loc:,} entities ({test_loc_pct:.1f}%)                                 │
  └─────────────────────────────────────────────────────────────────────┘

  KEY IMPROVEMENTS:
  ✓ Training now has Location entities: {train_loc:,} (was 0!)
  ✓ Dev and Test have similar Location %: {dev_loc_pct:.1f}% vs {test_loc_pct:.1f}%
  ✓ Location metrics will be statistically comparable
""")
    
    # Verify balance
    loc_diff = abs(dev_loc_pct - test_loc_pct)
    if loc_diff < 5:
        print(f"  ✓ BALANCED: Dev/Test Location difference is only {loc_diff:.1f}%")
    else:
        print(f"  ⚠ WARNING: Dev/Test Location difference is {loc_diff:.1f}%")
    
    print("\n" + "=" * 70)
    print("Done! Use these splits with your training script.")
    print("=" * 70)


if __name__ == "__main__":
    main()