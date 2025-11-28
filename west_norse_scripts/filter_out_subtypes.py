#!/usr/bin/env python3
"""
Script to remove entities with specific subtypes (ROL, NIC) from a JSON file.
"""

import json

# ============================================
# CONFIGURE THESE PATHS
# ============================================
INPUT_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/olafs_saga_helga/olafs_entities_norm_nested.json"
OUTPUT_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/olafs_saga_helga/olafs_entities_norm_flattened.json"
# ============================================

# Subtypes to remove
SUBTYPES_TO_REMOVE = {"ROL", "NIC"}


def main():
    # Load the JSON file
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        entities = json.load(f)

    # Count before filtering
    original_count = len(entities)

    # Filter out entities with ROL or NIC subtypes
    filtered_entities = [
        entity for entity in entities
        if entity.get("subtype") not in SUBTYPES_TO_REMOVE
    ]

    # Count after filtering
    filtered_count = len(filtered_entities)
    removed_count = original_count - filtered_count

    # Save the filtered entities
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(filtered_entities, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"Original entities: {original_count}")
    print(f"Removed entities (ROL/NIC): {removed_count}")
    print(f"Remaining entities: {filtered_count}")
    print(f"Output saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()