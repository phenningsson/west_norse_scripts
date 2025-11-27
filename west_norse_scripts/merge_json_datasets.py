#!/usr/bin/env python3
"""Combine multiple JSON dataset files into a single JSON array file."""

import json

# === CONFIGURE FILES HERE ===
INPUT_FILES = [
    "flat_ner/menota/laknisbok/laknisbok_flat_ner_dataset.json",
    "flat_ner/menota/am162bkfol/am162bk_flat_ner_dataset.json",
    "flat_ner/menota/am162b0fol/am162b0_flat_ner_dataset.json",
    "flat_ner/menota/olafs_saga_helga/olafs_saga_helga_flat_ner_dataset.json",
    "flat_ner/menota/islendingabok/islendingabok_flat_ner_dataset.json",
    "flat_ner/menota/alexanders_saga/alexanders_saga_flat_ner_dataset.json",
    "flat_ner/menota/drauma_jons/drauma_jons_flat_ner_dataset.json",
    "flat_ner/menota/wormianus/wormianus_flat_ner_dataset.json",
    "flat_ner/ihpc/1210_jartein/1210_jartein_flat_ner_dataset.json",
    "flat_ner/ihpc/1210_thorlakur/1210_thorlakur_flat_ner_dataset.json",
    "flat_ner/ihpc/1250_sturlunga/1250_sturlunga_flat_ner_dataset.json",
    "flat_ner/ihpc/1250_thetubrot/1250_thetubrot_flat_ner_dataset.json",
    "flat_ner/ihpc/1260_jomsvikingar/1260_jomsvikingar_flat_ner_dataset.json",
    "flat_ner/ihpc/1275_morkin/1275_morkin_flat_ner_dataset.json"
]

OUTPUT_FILE = "large_flat_ner_dataset.json"
# ============================

combined_data = []

for file_path in INPUT_FILES:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        combined_data.extend(data)
    else:
        combined_data.append(data)
    
    print(f"Added: {file_path}")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(combined_data, f, ensure_ascii=False, indent=2)

print(f"Combined {len(combined_data)} items into: {OUTPUT_FILE}")