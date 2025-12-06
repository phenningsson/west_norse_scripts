#!/usr/bin/env python3
"""
Convert entity annotations + normalized text to CoNLL/BIO format.
Compatible with MIM-GOLD-NER format for NER training.

CRITICAL: Tags ALL occurrences of each entity in the text,
not just the number of times it appears in the JSON annotations.

Output format:
TOKEN\tLABEL
- B-Person / I-Person for person names
- B-Location / I-Location for locations  
- O for outside any entity
- Empty line between sentences
"""

import json
import re
from pathlib import Path
from typing import List, Tuple, Dict
import unicodedata


def fix_mojibake(text: str) -> str:
    """
    Fix UTF-8 text that was incorrectly decoded as latin-1 (mojibake).
    E.g., "HlÃ­Ã°arenda" -> "Hlíðarenda"
    """
    try:
        # Try to fix double-encoded UTF-8
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text


def normalize_unicode(text: str) -> str:
    """
    Normalize text to NFC (composed) form.
    
    This converts decomposed characters (NFD) like:
      'o' + combining acute (U+0301) -> 'ó' (U+00F3)
      'o' + combining ogonek (U+0328) -> 'ǫ' (U+01EB)
    
    This ensures consistent matching between text and entities.
    """
    return unicodedata.normalize('NFC', text)


def normalize_text(text: str) -> str:
    """Normalize text for consistent matching."""
    text = normalize_unicode(text)
    return text


def tokenize_sentence(sentence: str) -> List[str]:
    """
    Simple tokenizer that splits on whitespace and separates punctuation.
    Handles Old Icelandic text appropriately.
    """
    raw_tokens = sentence.split()
    tokens = []
    
    for token in raw_tokens:
        # Separate leading punctuation
        while token and token[0] in '([«"\'':
            tokens.append(token[0])
            token = token[1:]
        
        # Separate trailing punctuation
        trailing = []
        while token and token[-1] in '.,;:!?)»"\'':
            trailing.insert(0, token[-1])
            token = token[:-1]
        
        if token:
            tokens.append(token)
        tokens.extend(trailing)
    
    return tokens


def load_entities(json_path: str) -> List[Dict]:
    """Load and fix entities from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)
    
    # Fix mojibake and normalize Unicode in entity texts
    fixed_entities = []
    for ent in entities:
        text = ent['norm_text']
        
        # First try mojibake fix (for latin-1 encoded files)
        text = fix_mojibake(text)
        
        # Then normalize to NFC (for NFD encoded files like Menota)
        text = normalize_unicode(text)
        
        fixed_ent = {
            'norm_text': text,
            'label': ent['label']
        }
        fixed_entities.append(fixed_ent)
    
    return fixed_entities


def load_text(text_path: str) -> str:
    """Load normalized text file and convert to NFC Unicode."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return normalize_unicode(text)


def map_label(label: str) -> str:
    """Map entity labels to MIM-GOLD-NER format."""
    label_map = {
        'PER': 'Person',
        'LOC': 'Location',
        'ORG': 'Organization',
        'MISC': 'Miscellaneous',
    }
    return label_map.get(label.upper(), label)


def find_entity_in_tokens(
    entity_text: str, 
    tokens: List[str], 
    start_idx: int = 0
) -> Tuple[int, int]:
    """
    Find entity span in token list.
    Returns (start_idx, end_idx) or (-1, -1) if not found.
    
    Handles multi-token entities, case variations, and dashed compounds.
    """
    entity_tokens = entity_text.split()
    
    # Try exact match first
    for i in range(start_idx, len(tokens) - len(entity_tokens) + 1):
        match = True
        for j, et in enumerate(entity_tokens):
            if tokens[i + j] != et:
                match = False
                break
        if match:
            return (i, i + len(entity_tokens))
    
    # Try case-insensitive match
    entity_tokens_lower = [t.lower() for t in entity_tokens]
    for i in range(start_idx, len(tokens) - len(entity_tokens) + 1):
        match = True
        for j, et in enumerate(entity_tokens_lower):
            if tokens[i + j].lower() != et:
                match = False
                break
        if match:
            return (i, i + len(entity_tokens))
    
    # Try matching dashed compounds
    if '-' in entity_text:
        dash_parts = re.split(r'-', entity_text)
        dash_parts = [p.strip() for p in dash_parts if p.strip()]
        
        if len(dash_parts) >= 2:
            for i in range(start_idx, len(tokens)):
                if tokens[i].lower() == dash_parts[0].lower():
                    match_end = try_match_dashed_entity(tokens, i, dash_parts)
                    if match_end != -1:
                        return (i, match_end)
    
    # Try single token match
    if len(entity_tokens) == 1:
        for i in range(start_idx, len(tokens)):
            if tokens[i].lower() == entity_tokens[0].lower():
                return (i, i + 1)
    
    return (-1, -1)


def try_match_dashed_entity(tokens: List[str], start: int, dash_parts: List[str]) -> int:
    """
    Try to match a dashed entity starting at position start.
    Returns end index if matched, -1 otherwise.
    """
    pos = start
    part_idx = 0
    
    while part_idx < len(dash_parts) and pos < len(tokens):
        current_token = tokens[pos].lower()
        expected_part = dash_parts[part_idx].lower()
        
        if current_token == expected_part:
            pos += 1
            part_idx += 1
            if pos < len(tokens) and tokens[pos] == '-':
                pos += 1
        elif current_token.rstrip('-') == expected_part:
            pos += 1
            part_idx += 1
        elif current_token.startswith(expected_part + '-'):
            remaining = current_token[len(expected_part)+1:]
            remaining_parts_match = True
            for remaining_part in dash_parts[part_idx+1:]:
                if remaining.startswith(remaining_part.lower()):
                    remaining = remaining[len(remaining_part):]
                    if remaining.startswith('-'):
                        remaining = remaining[1:]
                elif remaining == remaining_part.lower():
                    remaining = ''
                else:
                    remaining_parts_match = False
                    break
            
            if remaining_parts_match and remaining == '':
                return pos + 1
            else:
                return -1
        elif current_token == '-'.join(p.lower() for p in dash_parts):
            return pos + 1
        else:
            return -1
    
    if part_idx == len(dash_parts):
        return pos
    return -1


def align_entities_to_tokens(
    tokens: List[str],
    entities: List[Dict],
    text: str
) -> Tuple[List[str], List[Tuple[str, str, int]]]:
    """
    Align entities to tokens and return BIO labels.
    
    CRITICAL FIX: Tags ALL occurrences of each entity in the text,
    not just the number of times it appears in the JSON annotations.
    
    This ensures every instance of "Haraldur" gets tagged as B-Person,
    even if "Haraldur" only appears once in the entity list.
    
    Returns:
        labels: List of BIO labels for each token
        matched_entities: List of (entity_text, label, token_start_idx) for matched entities
    """
    labels = ['O'] * len(tokens)
    matched_entities = []
    
    # Get unique entities (we only need one entry per unique entity text + label)
    unique_entities = {}
    for ent in entities:
        key = (ent['norm_text'], ent['label'])
        if key not in unique_entities:
            unique_entities[key] = ent
    
    # Sort by entity length (number of tokens) - LONGEST FIRST
    # This ensures multi-word entities like "Haraldur Gormsson" get matched
    # before their components "Haraldur" and "Gormsson"
    sorted_entities = sorted(
        unique_entities.items(),
        key=lambda x: len(x[0][0].split()),
        reverse=True  # Longest first
    )
    
    # For each unique entity, find ALL occurrences in the text - NO LIMIT!
    for (ent_text, ent_label), ent in sorted_entities:
        label = map_label(ent_label)
        
        # Find ALL occurrences - keep searching until no more found
        search_start = 0
        
        while search_start < len(tokens):
            start, end = find_entity_in_tokens(ent_text, tokens, search_start)
            
            if start == -1:
                break  # No more occurrences found
            
            # Check if ANY token in the span is already labeled
            # (prevents overwriting longer entities with shorter ones)
            span_is_free = all(labels[i] == 'O' for i in range(start, end))
            
            if span_is_free:
                # Apply BIO labels
                labels[start] = f'B-{label}'
                for i in range(start + 1, end):
                    labels[i] = f'I-{label}'
                matched_entities.append((ent_text, ent_label, start))
            
            # Always move forward to find next occurrence
            search_start = start + 1
    
    return labels, matched_entities


def convert_to_conll(
    text_path: str,
    entities_path: str,
    output_path: str
) -> Tuple[int, int, int, int]:
    """
    Convert text + entities to CoNLL format.
    
    Returns: (num_sentences, num_tokens, num_entities_found, num_unique_entities)
    """
    # Load data
    text = load_text(text_path)
    text = normalize_text(text)
    entities = load_entities(entities_path)
    
    # Split into sentences (by newlines)
    sentences = re.split(r'\n+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Tokenize entire text first, keeping sentence boundaries
    all_tokens = []
    sentence_boundaries = []
    
    for sentence in sentences:
        tokens = tokenize_sentence(sentence)
        if not tokens:
            continue
        start_idx = len(all_tokens)
        all_tokens.extend(tokens)
        sentence_boundaries.append((start_idx, len(all_tokens)))
    
    # Align entities to ALL tokens at once
    labels, matched_entities = align_entities_to_tokens(all_tokens, entities, text)
    
    # Count found entities (number of B- tags)
    entities_found = sum(1 for label in labels if label.startswith('B-'))
    
    # Count unique entities in the JSON
    unique_in_json = len(set((e['norm_text'], e['label']) for e in entities))
    
    # Report statistics
    print(f"\n  Entity tagging results:")
    print(f"    Unique entities in JSON: {unique_in_json}")
    print(f"    Total occurrences tagged in text: {entities_found}")
    
    # Check for any entities that weren't found at all
    found_entity_texts = set(m[0].lower() for m in matched_entities)
    not_found = []
    for ent in entities:
        if ent['norm_text'].lower() not in found_entity_texts:
            key = (ent['norm_text'], ent['label'])
            if key not in [(n[0], n[1]) for n in not_found]:
                not_found.append((ent['norm_text'], ent['label']))
    
    if not_found:
        print(f"\n  Entities not found in text ({len(not_found)}):")
        for ent_text, ent_label in not_found[:20]:  # Show first 20
            print(f"    '{ent_text}' ({ent_label})")
        if len(not_found) > 20:
            print(f"    ... and {len(not_found) - 20} more")
    
    # Write output with sentence boundaries
    all_lines = []
    for start, end in sentence_boundaries:
        for i in range(start, end):
            all_lines.append(f'{all_tokens[i]}\t{labels[i]}')
        all_lines.append('')  # Empty line between sentences
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_lines))
    
    return len(sentences), len(all_tokens), entities_found, unique_in_json


def main():
    import sys
    
    # File paths 
    text_path = '/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/laknisbok/laknisbok_final_text.txt'
    entities_path = '/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/menota/laknisbok/laknisbok_entities_dipl_flat.json'
    output_path = '/Users/phenningsson/Downloads/west_norse_scripts/menota_conll_diplomatic/diplomatic_conll_laknisbok'

    # Allow command line arguments
    if len(sys.argv) >= 4:
        text_path = sys.argv[1]
        entities_path = sys.argv[2]
        output_path = sys.argv[3]
    
    print(f"Converting:")
    print(f"  Text: {text_path}")
    print(f"  Entities: {entities_path}")
    print(f"  Output: {output_path}")
    print()
    
    # Load and display some entities to verify encoding fix
    entities = load_entities(entities_path)
    print("Sample entities (after encoding fix):")
    for ent in entities[:10]:
        print(f"  {ent['norm_text']:20s} -> {ent['label']}")
    
    # Convert
    num_sentences, num_tokens, num_entities, num_unique = convert_to_conll(
        text_path, entities_path, output_path
    )
    
    print(f"\nFinal Results:")
    print(f"  Sentences: {num_sentences}")
    print(f"  Tokens: {num_tokens}")
    print(f"  Unique entities in JSON: {num_unique}")
    print(f"  Total entity occurrences tagged: {num_entities}")
    print(f"  Output: {output_path}")


if __name__ == '__main__':
    main()