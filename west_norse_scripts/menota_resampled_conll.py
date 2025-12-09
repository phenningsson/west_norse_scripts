#!/usr/bin/env python3
"""
Resample Menota Training Set (sCR method)

This script applies sentence-level resampling (Wang & Wang 2022) to the 
Menota training set to boost underrepresented Location entities.
"""

import os
import math
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_normalised/menota_only/menota_only.txt"
OUTPUT_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/conll_experiments_normalised/menota_resampl/menota_train_resampled.txt"
RESAMPLING_METHOD = 'sCR'
RANDOM_SEED = 42

# =============================================================================
# PARSING FUNCTIONS
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


def count_entity_tokens(sentence: List[Tuple[str, str]]) -> Dict[str, int]:
    """Count all entity tokens (B- and I-) by type."""
    counts = defaultdict(int)
    for token, label in sentence:
        if label.startswith('B-') or label.startswith('I-'):
            counts[label[2:]] += 1
    return dict(counts)


def count_entities(sentences: List[List[Tuple[str, str]]]) -> Dict[str, int]:
    """Count B- tags (entity starts) by type."""
    counts = defaultdict(int)
    for sentence in sentences:
        for token, label in sentence:
            if label.startswith('B-'):
                counts[label[2:]] += 1
    return dict(counts)


def get_corpus_stats(sentences: List[List[Tuple[str, str]]]) -> Dict:
    """Get comprehensive corpus statistics."""
    total_tokens = 0
    token_counts = defaultdict(int)
    
    for sentence in sentences:
        total_tokens += len(sentence)
        for token, label in sentence:
            if label.startswith('B-') or label.startswith('I-'):
                etype = label[2:]
                token_counts[etype] += 1
    
    # Compute rareness (for resampling)
    rareness = {}
    for etype, count in token_counts.items():
        prob = count / total_tokens if total_tokens > 0 else 0
        rareness[etype] = -math.log2(prob) if prob > 0 else 0
    
    return {
        'total_tokens': total_tokens,
        'token_counts': dict(token_counts),
        'rareness': rareness,
    }


def compute_resampling_factor(
    sentence: List[Tuple[str, str]],
    stats: Dict,
    method: str
) -> int:
    """Compute resampling factor for a sentence using sCR method."""
    entity_tokens = count_entity_tokens(sentence)
    rareness = stats['rareness']
    
    if not entity_tokens:
        return 1
    
    if method == 'sCR':
        # Equation 2: Count weighted by rareness, with sqrt dampening
        weighted = sum(rareness.get(t, 1) * c for t, c in entity_tokens.items())
        return 1 + math.ceil(math.sqrt(weighted)) if weighted > 0 else 1
    
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
    print("Resample Menota Training Set (sCR)")
    print("=" * 70)
    
    # Load original training set
    print("\n1. Loading original Menota training set...")
    sentences = parse_conll_file(INPUT_FILE)
    original_entities = count_entities(sentences)
    
    print(f"   Sentences: {len(sentences)}")
    print(f"   Entities:")
    for etype, count in sorted(original_entities.items()):
        print(f"     {etype}: {count}")
    
    # Apply resampling
    print(f"\n2. Applying {RESAMPLING_METHOD} resampling...")
    resampled, stats = resample_sentences(sentences, RESAMPLING_METHOD, RANDOM_SEED)
    
    print(f"\n   Rareness scores:")
    for etype, rare in sorted(stats['rareness'].items(), key=lambda x: -x[1]):
        print(f"     {etype}: {rare:.2f}")
    
    print(f"\n   Resampling results:")
    print(f"     Original:  {stats['original']:,} sentences")
    print(f"     Resampled: {stats['resampled']:,} sentences")
    print(f"     Expansion: {stats['expansion']:.2f}x")
    
    print(f"\n   Factor distribution:")
    for factor, count in sorted(stats['factors'].items()):
        pct = 100 * count / stats['original']
        bar = '█' * int(pct / 2)
        print(f"     {factor}x: {count:5} ({pct:5.1f}%) {bar}")
    
    # Count entities in resampled set
    resampled_entities = count_entities(resampled)
    
    print(f"\n3. Entity comparison (Original → Resampled):")
    for etype in sorted(set(original_entities.keys()) | set(resampled_entities.keys())):
        orig = original_entities.get(etype, 0)
        resamp = resampled_entities.get(etype, 0)
        ratio = resamp / orig if orig > 0 else 0
        print(f"     {etype}: {orig} → {resamp} ({ratio:.2f}x)")
    
    # Write output
    print(f"\n4. Writing resampled training set...")
    write_conll_file(OUTPUT_FILE, resampled)
    print(f"   Written: {OUTPUT_FILE}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    orig_loc_pct = 100 * original_entities.get('Location', 0) / sum(original_entities.values())
    resamp_loc_pct = 100 * resampled_entities.get('Location', 0) / sum(resampled_entities.values())
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ ORIGINAL MENOTA TRAINING SET                                        │
  │   Sentences: {stats['original']:,}                                              │
  │   Person: {original_entities.get('Person', 0):,}, Location: {original_entities.get('Location', 0)} ({orig_loc_pct:.1f}% of entities)           │
  ├─────────────────────────────────────────────────────────────────────┤
  │ RESAMPLED MENOTA TRAINING SET (sCR)                                 │
  │   Sentences: {stats['resampled']:,}                                             │
  │   Person: {resampled_entities.get('Person', 0):,}, Location: {resampled_entities.get('Location', 0)} ({resamp_loc_pct:.1f}% of entities)          │
  │   Expansion: {stats['expansion']:.2f}x                                                │
  ├─────────────────────────────────────────────────────────────────────┤
  │ LOCATION BOOST                                                      │
  │   Original:  {original_entities.get('Location', 0)} Location entities                                   │
  │   Resampled: {resampled_entities.get('Location', 0)} Location entities                                  │
  │   Boost:     {resampled_entities.get('Location', 0) / original_entities.get('Location', 1):.2f}x more Location training signal                      │
  └─────────────────────────────────────────────────────────────────────┘

  You now have TWO Menota training experiments:
  
  1. menota_train.txt           - Original distribution (baseline)
  2. menota_train_resampled.txt - sCR resampled (Location boosted)
  
  Both use the SAME dev/test splits for fair comparison.
""")


if __name__ == "__main__":
    main()