import pandas as pd
import json

# Load the CSV file
csv_file = 'wormianus/short_wormianus_names.csv'
df = pd.read_csv(csv_file)

# Filter out entities labeled as 'WRK'
filtered_df = df[df['entity_type'] != 'WRK']
new_filtered_df = filtered_df
new_filtered_df = new_filtered_df[df['entity_type'] != "OTH"]

# Create the JSON structure
json_data = [
    {"dipl_text": row['entity'], "label": row['entity_type']}
    for _, row in new_filtered_df.iterrows()
]

# Save the JSON data to a file
output_file = 'dipl_wormianus_entities.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"JSON file saved as {output_file}")
