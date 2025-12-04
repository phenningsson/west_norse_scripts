#!/usr/bin/env python3
"""
Filter MIM-GOLD-NER CoNLL files to keep only Person and Location entities.
All other entity types (Organization, Date, Time, Money, Percent, Miscellaneous) 
are converted to 'O' (outside).

Usage:
    # Filter a single file
    python filter_mim_gold.py input.txt -o output.txt
    
    # Filter multiple files (creates filtered versions with _filtered suffix)
    python filter_mim_gold.py file1.txt file2.txt file3.txt
    
    # Filter all .txt files in a directory
    python filter_mim_gold.py --dir /path/to/mim_gold_ner/ --output-dir /path/to/filtered/
    
python3 /Users/phenningsson/Downloads/west_norse_scripts/filter_mim_data.py --dir /Users/phenningsson/Downloads/west_norse_scripts/conll_format/mim_gold_ner/all --output-dir /Users/phenningsson/Downloads/west_norse_scripts/conll_format/mim_gold_ner/filtered
    
    # Preview changes without writing (dry run)
    python filter_mim_gold.py input.txt --dry-run
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Set
from collections import Counter


# Entity types to KEEP (all others will be converted to 'O')
KEEP_ENTITY_TYPES = {'Person', 'Location'}


def filter_conll_file(
    input_path: str,
    output_path: str = None,
    keep_types: Set[str] = None,
    dry_run: bool = False
) -> dict:
    """
    Filter a CoNLL file to keep only specified entity types.
    
    Args:
        input_path: Path to input CoNLL file
        output_path: Path to output file (None = auto-generate with _filtered suffix)
        keep_types: Set of entity types to keep (default: Person, Location)
        dry_run: If True, don't write output, just report statistics
    
    Returns:
        Dictionary with statistics
    """
    if keep_types is None:
        keep_types = KEEP_ENTITY_TYPES
    
    if output_path is None:
        # Auto-generate output filename
        input_p = Path(input_path)
        output_path = str(input_p.parent / f"{input_p.stem}_filtered{input_p.suffix}")
    
    # Statistics
    stats = {
        'input_file': input_path,
        'output_file': output_path,
        'total_tokens': 0,
        'total_sentences': 0,
        'original_entities': Counter(),
        'kept_entities': Counter(),
        'removed_entities': Counter(),
    }
    
    output_lines = []
    current_sentence_tokens = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            
            if line == '':
                # Empty line = sentence boundary
                output_lines.append('')
                if current_sentence_tokens > 0:
                    stats['total_sentences'] += 1
                current_sentence_tokens = 0
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    token, label = parts[0], parts[1]
                    stats['total_tokens'] += 1
                    current_sentence_tokens += 1
                    
                    if label == 'O':
                        # Already outside - keep as is
                        output_lines.append(f"{token}\t{label}")
                    elif label.startswith(('B-', 'I-')):
                        # Entity tag - check if we keep it
                        prefix = label[:2]  # 'B-' or 'I-'
                        entity_type = label[2:]  # 'Person', 'Location', etc.
                        
                        stats['original_entities'][entity_type] += 1
                        
                        if entity_type in keep_types:
                            # Keep this entity type
                            output_lines.append(f"{token}\t{label}")
                            stats['kept_entities'][entity_type] += 1
                        else:
                            # Remove this entity type (convert to O)
                            output_lines.append(f"{token}\tO")
                            stats['removed_entities'][entity_type] += 1
                    else:
                        # Unknown label format - keep as is
                        output_lines.append(line)
                else:
                    # Malformed line - keep as is
                    output_lines.append(line)
    
    # Count last sentence if file doesn't end with empty line
    if current_sentence_tokens > 0:
        stats['total_sentences'] += 1
    
    # Write output
    if not dry_run:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
            if output_lines and output_lines[-1] != '':
                f.write('\n')
    
    return stats


def print_stats(stats: dict, verbose: bool = True):
    """Print filtering statistics."""
    print(f"\n{'='*60}")
    print(f"File: {stats['input_file']}")
    print(f"{'='*60}")
    print(f"  Sentences: {stats['total_sentences']:,}")
    print(f"  Tokens: {stats['total_tokens']:,}")
    
    print(f"\n  Original entities:")
    for etype, count in sorted(stats['original_entities'].items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count:,}")
    
    print(f"\n  After filtering (kept):")
    for etype, count in sorted(stats['kept_entities'].items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count:,}")
    
    if stats['removed_entities']:
        print(f"\n  Removed (converted to O):")
        for etype, count in sorted(stats['removed_entities'].items(), key=lambda x: -x[1]):
            print(f"    {etype}: {count:,}")
    
    total_original = sum(stats['original_entities'].values())
    total_kept = sum(stats['kept_entities'].values())
    total_removed = sum(stats['removed_entities'].values())
    
    print(f"\n  Summary:")
    print(f"    Original entities: {total_original:,}")
    print(f"    Kept entities: {total_kept:,}")
    print(f"    Removed entities: {total_removed:,}")
    print(f"    Output: {stats['output_file']}")


def process_directory(
    input_dir: str,
    output_dir: str = None,
    keep_types: Set[str] = None,
    dry_run: bool = False
) -> List[dict]:
    """Process all .txt files in a directory."""
    input_path = Path(input_dir)
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path
    
    all_stats = []
    txt_files = list(input_path.glob('*.txt'))
    
    print(f"Found {len(txt_files)} .txt files in {input_dir}")
    
    for input_file in sorted(txt_files):
        output_file = output_path / f"{input_file.stem}_filtered.txt"
        stats = filter_conll_file(
            str(input_file),
            str(output_file),
            keep_types,
            dry_run
        )
        all_stats.append(stats)
        print_stats(stats)
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description='Filter MIM-GOLD-NER files to keep only Person and Location entities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Filter a single file
    python filter_mim_gold.py mbl.txt -o mbl_filtered.txt
    
    # Filter multiple files (auto-generates output names)
    python filter_mim_gold.py mbl.txt fbl.txt books.txt
    
    # Filter all files in a directory
    python filter_mim_gold.py --dir ./mim_gold_ner/ --output-dir ./filtered/
    
    # Preview without writing files
    python filter_mim_gold.py mbl.txt --dry-run
    
    # Keep different entity types
    python filter_mim_gold.py mbl.txt --keep Person Location Organization
        """
    )
    
    parser.add_argument('files', nargs='*', help='Input CoNLL files to filter')
    parser.add_argument('-o', '--output', help='Output file (for single input file)')
    parser.add_argument('--dir', help='Process all .txt files in directory')
    parser.add_argument('--output-dir', help='Output directory (with --dir)')
    parser.add_argument('--keep', nargs='+', default=['Person', 'Location'],
                        help='Entity types to keep (default: Person Location)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show statistics without writing files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    keep_types = set(args.keep)
    print(f"Keeping entity types: {keep_types}")
    print(f"All other types will be converted to 'O'")
    
    if args.dir:
        # Process directory
        all_stats = process_directory(
            args.dir,
            args.output_dir,
            keep_types,
            args.dry_run
        )
        
        # Print combined summary
        print(f"\n{'='*60}")
        print("COMBINED SUMMARY")
        print(f"{'='*60}")
        
        total_original = Counter()
        total_kept = Counter()
        total_removed = Counter()
        total_tokens = 0
        total_sentences = 0
        
        for stats in all_stats:
            total_tokens += stats['total_tokens']
            total_sentences += stats['total_sentences']
            total_original.update(stats['original_entities'])
            total_kept.update(stats['kept_entities'])
            total_removed.update(stats['removed_entities'])
        
        print(f"  Files processed: {len(all_stats)}")
        print(f"  Total sentences: {total_sentences:,}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total original entities: {sum(total_original.values()):,}")
        print(f"  Total kept entities: {sum(total_kept.values()):,}")
        print(f"  Total removed entities: {sum(total_removed.values()):,}")
    
    elif args.files:
        # Process individual files
        for input_file in args.files:
            if not os.path.exists(input_file):
                print(f"Warning: File not found: {input_file}")
                continue
            
            output_file = args.output if (args.output and len(args.files) == 1) else None
            stats = filter_conll_file(input_file, output_file, keep_types, args.dry_run)
            print_stats(stats)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()