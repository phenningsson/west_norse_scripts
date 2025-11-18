import json
from collections import defaultdict

def analyze_entity_file(entity_file):
    """Analyze entity file and return counts by type and subtype."""
    with open(entity_file, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    # Initialize counters
    total_entities = len(entities)
    type_counts = defaultdict(int)
    subtype_counts = defaultdict(lambda: defaultdict(int))

    for entity in entities:
        # Count by main type (label)
        entity_type = entity.get('label', 'UNKNOWN')
        type_counts[entity_type] += 1

        # Count by subtype if present
        subtype = entity.get('subtype')
        if subtype:
            subtype_counts[entity_type][subtype] += 1

    return {
        'total_entities': total_entities,
        'type_counts': dict(type_counts),
        'subtype_counts': {k: dict(v) for k, v in subtype_counts.items()}
    }

def analyze_word_file(word_file):
    """Analyze word file and return total word count."""
    with open(word_file, 'r', encoding='utf-8') as f:
        words = json.load(f)

    return {
        'total_words': len(words)
    }

def print_analysis(entity_results, word_results):
    """Print the analysis results in a readable format."""
    print("=== ENTITY ANALYSIS ===")
    print(f"Total entities: {entity_results['total_entities']}")

    print("\nBy entity type:")
    for type_name, count in entity_results['type_counts'].items():
        print(f"  {type_name}: {count}")

    print("\nBy subtype:")
    for type_name, subtypes in entity_results['subtype_counts'].items():
        print(f"  {type_name}:")
        for subtype, count in subtypes.items():
            print(f"    {subtype}: {count}")

    print("\n=== WORD ANALYSIS ===")
    print(f"Total words: {word_results['total_words']}")

def main():
    entity_file = 'alexanders_saga/alexander_entities_with_positions.json'
    word_file = 'alexanders_saga/alexander_word_positions.json'

    # Analyze both files
    entity_results = analyze_entity_file(entity_file)
    word_results = analyze_word_file(word_file)

    # Print results
    print_analysis(entity_results, word_results)

if __name__ == "__main__":
    main()
