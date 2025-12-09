#!/usr/bin/env python3
"""
Extract normalized text from Menota TEI XML files.

This script extracts the content from <me:norm> tags within the <body>
element of Menota-encoded XML files and outputs clean, readable text.

Input: Menota TEI XML file (can be .xml or .txt extension)
Output: Clean normalized text file
"""

import re
from pathlib import Path


# ============================================================
# CONFIGURATION - 
# ============================================================

# Input XML/TXT file
INPUT_FILE = "//Users/phenningsson/Downloads/west_norse_scripts/AM-36-fol.xml.txt"

# Output text file
OUTPUT_FILE = "/Users/phenningsson/Downloads/west_norse_scripts/heimskringla_2_dipl.txt"

# ============================================================


# Entity replacements for common Menota entities
ENTITY_MAP = {
    '&horbar;': '-',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&apos;': "'",
    '&quot;': '"',
}


def extract_norm_text(content: str) -> str:
    """
    Extract normalized text from Menota XML content.
    
    Args:
        content: Raw XML content as string
        
    Returns:
        Clean normalized text
    """
    # Try multiple patterns to find body content (handles namespaces)
    body_patterns = [
        r'<body[^>]*>(.*?)</body>',
        r'<tei:body[^>]*>(.*?)</tei:body>',
        r'<TEI:body[^>]*>(.*?)</TEI:body>',
    ]
    
    body_content = None
    for pattern in body_patterns:
        body_match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if body_match:
            body_content = body_match.group(1)
            break
    
    if not body_content:
        print("Warning: No <body> element found")
        print(f"File size: {len(content)} characters")
        # Show what tags exist
        tags = set(re.findall(r'<(\w+:?\w*)[^>]*>', content[:5000]))
        print(f"Tags found in first 5000 chars: {sorted(tags)[:20]}")
        return ""
    
    print(f"Body content found: {len(body_content)} characters")
    
    # Handle self-closing <me:norm/> tags by removing them first
    body_content = re.sub(r'<me:dipl\s*/>', '', body_content)
    
    # Find all <me:norm> content
    # Simple pattern: get everything between <me:norm> and </me:norm>
    norm_pattern = r'<me:dipl[^>]*>(.*?)</me:dipl>'
    
    matches = list(re.finditer(norm_pattern, body_content, re.DOTALL))
    print(f"Found {len(matches)} <me:dipl> matches")
    
    tokens = []
    for match in matches:
        text = match.group(1)
        
        # Remove any remaining XML tags (like <supplied>, <ex>, etc.)
        text = re.sub(r'<[^>]+>', '', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        # Replace known entities
        for entity, char in ENTITY_MAP.items():
            text = text.replace(entity, char)
        
        # Remove any remaining unrecognized entities
        text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)
        
        # Handle numeric entities
        text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        
        text = text.strip()
        if text:
            tokens.append(text)
    
    # Join tokens with spaces
    text = ' '.join(tokens)
    
    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure space before punctuation
    text = re.sub(r'(\S)([.,:;!?…―\-–—])', r'\1 \2', text)
    
    # Ensure space after punctuation
    text = re.sub(r'([.,:;!?…―\-–—])(\S)', r'\1 \2', text)
    
    # Add linebreak after sentence-ending punctuation followed by closing quote
    # This must come BEFORE the general sentence-ending rule
    text = re.sub(r'([.!?…]) " ', r'\1 "\n', text)
    
    # Add linebreak after sentence-ending punctuation (. ! ? …) NOT followed by closing quote
    text = re.sub(r'([.!?…]) (?!")', r'\1\n', text)
    
    # Clean up any double spaces that may have been created
    text = re.sub(r' +', ' ', text)
    
    # Clean up any spaces at the beginning of lines
    text = re.sub(r'\n ', r'\n', text)
    
    return text.strip()


def process_file(input_path: str, output_path: str):
    """
    Process a single XML file and save extracted normalized text.
    
    Args:
        input_path: Path to the input XML file
        output_path: Path to the output text file
    """
    print(f"Reading: {input_path}")
    
    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    
    # Extract text
    text = extract_norm_text(content)
    
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    num_words = len(text.split())
    print(f"Output: {output_path}")
    print(f"Words: {num_words:,}")


if __name__ == '__main__':
    process_file(INPUT_FILE, OUTPUT_FILE)