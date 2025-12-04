#!/usr/bin/env python3
"""Combine multiple JSON dataset files into a single JSON array file."""

import json

# === CONFIGURE FILES HERE ===
INPUT_FILES = [
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/dataset/laknisbok_normalised_flat_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/dataset/am162bk_normalised_flat_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/dataset/am162b0_normalised_flat_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/dataset/olafs_saga_normalised_flat_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/dataset/islendingabok_normalised_flat_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/menota_normalised/dataset/alexander_normalised_flat_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/ihpc/1210_jartein/1210_jartein_flat_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/ihpc/1210_thorlakur/1210_thorlakur_flat_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/ihpc/1250_sturlunga/1250_sturlunga_flat_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/ihpc/1250_thetubrot/1250_thetubrot_flat_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/ihpc/1260_jomsvikingar/1260_jomsvikingar_flat_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/flat_ner_diplomatic/ihpc/1275_morkin/1275_morkin_flat_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/adjudications_mim_ner_dataset_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/blog_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/books_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/emails_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/fbl_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/laws_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/mbl_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/radio_tv_news_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/school_essays_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/scienceweb_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/webmedia_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/websites_mim_ner_dataset.json",
    "/Users/phenningsson/Downloads/west_norse_scripts/mim_gold_ner/dataset/written-to-be-spoken_mim_ner_dataset.json"
]

OUTPUT_FILE = "largest_normalised_flat_ner_dataset.json"
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