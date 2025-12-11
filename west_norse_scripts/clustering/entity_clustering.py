#!/usr/bin/env python
"""
Entity Clustering v3 - With Interactive Editor
==============================================

Same clustering as v3, but generates an interactive HTML editor where you can:
- Drag entities between clusters
- Merge clusters together
- Split entities out to new clusters
- Mark entities as noise (unclustered)
- Export corrected clusters to JSON

Usage:
    python entity_clustering_v3_interactive.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = "/Users/phenningsson/icelandic_clustering/clustering/icebert-old-icelandic-v9"
ENTITIES_FILE = "/Users/phenningsson/icelandic_clustering/clustering/menota_flat_ner_dataset.json"
OUTPUT_DIR = "/Users/phenningsson/icelandic_clustering/clustering/clustering_results_v6_interactive"

# Hybrid distance weights - STRING FOCUSED
STRING_WEIGHT = 0.70
EMBEDDING_WEIGHT = 0.30

# Clustering parameters
MIN_CLUSTER_SIZE = 2
MIN_SAMPLES = 1
CLUSTER_SELECTION_EPSILON = 0.0
CLUSTER_SELECTION_METHOD = "leaf"

CLUSTER_BY_TYPE = True

# ============================================================================
# TEXT HANDLING (NO CHARACTER NORMALIZATION)
# ============================================================================

def normalize_text(text, aggressive=False):
    """Lowercase only - preserves all original characters."""
    return text.lower()

# ============================================================================
# STRING SIMILARITY METRICS
# ============================================================================

def jaro_winkler_similarity(s1, s2):
    """Jaro-Winkler similarity."""
    try:
        from Levenshtein import jaro_winkler
        return jaro_winkler(s1, s2)
    except ImportError:
        return jaro_similarity(s1, s2)

def jaro_similarity(s1, s2):
    """Basic Jaro similarity."""
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    match_distance = max(len1, len2) // 2 - 1
    match_distance = max(0, match_distance)
    
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    
    matches = 0
    transpositions = 0
    
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len2)
        
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break
    
    if matches == 0:
        return 0.0
    
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3
    return jaro

def normalized_levenshtein(s1, s2):
    """Normalized Levenshtein distance."""
    try:
        from Levenshtein import distance
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        return distance(s1, s2) / max_len
    except ImportError:
        return basic_levenshtein(s1, s2) / max(len(s1), len(s2), 1)

def basic_levenshtein(s1, s2):
    """Basic Levenshtein distance."""
    if len(s1) < len(s2):
        return basic_levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def common_prefix_length(s1, s2):
    """Length of common prefix."""
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

def combined_string_similarity(s1, s2):
    """Combined string similarity (no character normalization)."""
    s1_lower = s1.lower()
    s2_lower = s2.lower()
    
    if s1_lower == s2_lower:
        return 1.0
    
    jw_sim = jaro_winkler_similarity(s1_lower, s2_lower)
    lev_dist = normalized_levenshtein(s1_lower, s2_lower)
    lev_sim = 1.0 - lev_dist
    
    prefix_len = common_prefix_length(s1_lower, s2_lower)
    max_len = max(len(s1_lower), len(s2_lower))
    prefix_ratio = prefix_len / max_len if max_len > 0 else 0
    
    combined = 0.40 * jw_sim + 0.35 * lev_sim + 0.25 * prefix_ratio
    
    return combined

def string_distance(s1, s2):
    """String distance (0 = identical, 1 = completely different)."""
    return 1.0 - combined_string_similarity(s1, s2)

# ============================================================================
# ENTITY LOADING
# ============================================================================

def load_entities(filepath):
    """Load entities from NER dataset, including source metadata (work, id)."""
    entities_by_type = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        peek = f.read(10000)
        f.seek(0)
        
        if peek.strip().startswith('['):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
    
    for entry in data:
        text = entry.get("text", "")
        ent_types = entry.get("entity_types") or entry.get("ner_tags") or []
        ent_starts = entry.get("entity_start_chars") or entry.get("starts") or []
        ent_ends = entry.get("entity_end_chars") or entry.get("ends") or []
        
        # Get source metadata
        work = entry.get("work", "Unknown")
        doc_id = entry.get("id", None)
        
        for etype, start, end in zip(ent_types, ent_starts, ent_ends):
            entity_text = text[start:end]
            if entity_text.strip():
                entities_by_type[etype].append({
                    "text": entity_text,
                    "start": start,
                    "end": end,
                    "context": text[max(0, start-30):end+30],
                    "work": work,
                    "doc_id": doc_id
                })
    
    return entities_by_type

def deduplicate_entities(entities):
    """
    Deduplicate entities by lowercase text, preserving ALL occurrences' metadata.
    
    For each unique entity text, we keep:
    - frequency: total count across all texts
    - contexts: list of ALL contexts where it appears (with source info)
    - variants: different surface forms (e.g., "Gunnarr" vs "gunnarr")
    
    This preserves the information that the same entity appears in multiple places.
    """
    seen = {}
    unique = []
    
    for ent in entities:
        key = ent["text"].lower()
        
        # Create context object with source metadata
        context_obj = {
            "text": ent.get("context", ""),
            "work": ent.get("work", "Unknown"),
            "doc_id": ent.get("doc_id"),
            "start": ent.get("start"),
            "end": ent.get("end")
        }
        
        if key not in seen:
            # First occurrence - initialize with lists
            ent["frequency"] = 1
            ent["contexts"] = [context_obj]
            ent["variants"] = {ent["text"]}  # Track different surface forms
            seen[key] = len(unique)
            unique.append(ent)
        else:
            # Additional occurrence - accumulate metadata
            existing = unique[seen[key]]
            existing["frequency"] = existing.get("frequency", 1) + 1
            
            # Add context with source info (avoid exact duplicates)
            existing_contexts = existing.get("contexts", [])
            # Check if this exact context already exists
            is_duplicate = any(
                c.get("text") == context_obj["text"] and c.get("work") == context_obj["work"]
                for c in existing_contexts
            )
            if not is_duplicate:
                existing_contexts.append(context_obj)
                existing["contexts"] = existing_contexts
            
            # Track variant spellings (case differences, etc.)
            existing.setdefault("variants", set()).add(ent["text"])
    
    # Convert sets to lists for JSON serialization
    for ent in unique:
        if "variants" in ent:
            ent["variants"] = list(ent["variants"])
    
    return unique

# ============================================================================
# EMBEDDING COMPUTATION
# ============================================================================

def get_embeddings(texts, model_path):
    """Get embeddings for texts using IceBERT."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    
    embeddings = []
    batch_size = 32
    
    print(f"Computing embeddings for {len(texts)} entities...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=64, return_tensors="pt")
            outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# ============================================================================
# DISTANCE MATRIX AND CLUSTERING
# ============================================================================

def compute_hybrid_distance_matrix(entities, embeddings):
    """Compute hybrid distance matrix."""
    n = len(entities)
    texts = [e["text"] for e in entities]
    
    print(f"Computing distance matrix for {n} entities...")
    
    # String distance matrix
    string_dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = string_distance(texts[i], texts[j])
            string_dist[i, j] = d
            string_dist[j, i] = d
    
    # Embedding distance matrix (cosine)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    cosine_sim = np.dot(normalized, normalized.T)
    embedding_dist = 1.0 - cosine_sim
    embedding_dist = np.clip(embedding_dist, 0, 2)
    
    # Hybrid distance
    hybrid_dist = STRING_WEIGHT * string_dist + EMBEDDING_WEIGHT * embedding_dist
    np.fill_diagonal(hybrid_dist, 0)
    
    return hybrid_dist, string_dist

def cluster_entities(distance_matrix):
    """Cluster entities using HDBSCAN."""
    import hdbscan
    
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
    )
    labels = clusterer.fit_predict(distance_matrix)
    
    return labels

# ============================================================================
# INTERACTIVE HTML EDITOR
# ============================================================================

def create_interactive_editor(entities, labels, entity_type, output_dir):
    """Create interactive HTML editor for cluster curation."""
    
    # Organize into clusters
    clusters = defaultdict(list)
    noise = []
    
    for i, (ent, label) in enumerate(zip(entities, labels)):
        # Prepare contexts with source info (limit to first 20 for performance)
        contexts = ent.get("contexts", [])[:20]
        variants = ent.get("variants", [ent["text"]])
        
        # Get unique works for this entity
        works = list(set(c.get("work", "Unknown") for c in contexts if c.get("work")))
        
        ent_data = {
            "id": i,
            "text": ent["text"],
            "contexts": contexts,  # Now includes work and doc_id
            "variants": variants,
            "works": works,
            "frequency": ent.get("frequency", 1),
            "numSources": len(contexts)
        }
        if label >= 0:
            clusters[int(label)].append(ent_data)
        else:
            noise.append(ent_data)
    
    # Convert to list format for JSON
    clusters_list = [
        {"id": cid, "entities": ents} 
        for cid, ents in sorted(clusters.items())
    ]
    
    # Prepare initial data as JSON
    initial_data = {
        "entityType": entity_type,
        "clusters": clusters_list,
        "noise": noise,
        "nextClusterId": max(clusters.keys()) + 1 if clusters else 0
    }
    
    html = """<!DOCTYPE html>
<html>
<head>
    <title>""" + entity_type + """ Cluster Editor</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }
        
        .header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; padding: 15px; background: white;
            border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        h1 { margin: 0; color: #333; }
        
        .toolbar {
            display: flex; gap: 10px;
        }
        
        .btn {
            padding: 10px 20px; border: none; border-radius: 5px;
            cursor: pointer; font-size: 14px; font-weight: 500;
            transition: all 0.2s;
        }
        
        .btn-primary { background: #4CAF50; color: white; }
        .btn-primary:hover { background: #45a049; }
        .btn-secondary { background: #2196F3; color: white; }
        .btn-secondary:hover { background: #1976D2; }
        .btn-warning { background: #ff9800; color: white; }
        .btn-warning:hover { background: #f57c00; }
        .btn-danger { background: #f44336; color: white; }
        .btn-danger:hover { background: #d32f2f; }
        
        .stats {
            background: #e8f5e9; padding: 10px 15px; border-radius: 5px;
            margin-bottom: 20px; display: flex; gap: 20px; flex-wrap: wrap;
        }
        
        .stat { font-size: 14px; }
        .stat b { color: #2e7d32; }
        
        .search-container {
            margin-bottom: 20px;
        }
        
        .search-box {
            width: 100%; padding: 12px; font-size: 16px;
            border: 2px solid #ddd; border-radius: 8px;
            transition: border-color 0.2s;
        }
        
        .search-box:focus { outline: none; border-color: #4CAF50; }
        
        .main-container {
            display: flex; gap: 20px;
        }
        
        .clusters-panel {
            flex: 3; display: flex; flex-direction: column; gap: 15px;
        }
        
        .noise-panel {
            flex: 1; background: white; padding: 15px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-height: 80vh;
            overflow-y: auto; position: sticky; top: 20px;
        }
        
        .noise-panel h3 { margin-top: 0; color: #666; }
        
        .cluster-card {
            background: white; border-radius: 8px; padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #4CAF50;
        }
        
        .cluster-card.drag-over {
            border-color: #2196F3; background: #e3f2fd;
        }
        
        .cluster-card.selected {
            border-color: #ff9800; box-shadow: 0 0 0 2px #ff9800;
        }
        
        .cluster-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #eee;
        }
        
        .cluster-title {
            font-weight: 600; color: #333;
        }
        
        .cluster-actions {
            display: flex; gap: 5px;
        }
        
        .cluster-actions button {
            padding: 4px 8px; font-size: 12px; border: none;
            border-radius: 3px; cursor: pointer;
        }
        
        .btn-select { background: #fff3e0; color: #e65100; }
        .btn-select.active { background: #ff9800; color: white; }
        .btn-delete { background: #ffebee; color: #c62828; }
        
        .entity-list {
            display: flex; flex-wrap: wrap; gap: 8px;
            min-height: 40px;
        }
        
        .entity {
            background: #e8f5e9; padding: 6px 12px; border-radius: 20px;
            font-size: 14px; cursor: grab; user-select: none;
            display: flex; align-items: center; gap: 6px;
            transition: all 0.2s; border: 2px solid transparent;
        }
        
        .entity:hover { background: #c8e6c9; }
        .entity.dragging { opacity: 0.5; }
        .entity.selected { border-color: #ff9800; background: #fff3e0; }
        
        .entity .freq {
            background: #81c784; color: white; padding: 2px 6px;
            border-radius: 10px; font-size: 11px; font-weight: bold;
        }
        
        .entity .freq.high { background: #388e3c; }
        .entity .freq.very-high { background: #1b5e20; }
        
        .entity .works-badge {
            background: #e3f2fd; color: #1565c0; padding: 2px 6px;
            border-radius: 10px; font-size: 10px; font-weight: bold;
        }
        
        .entity .info-btn {
            color: #666; cursor: pointer; font-size: 12px;
            margin-left: 2px; padding: 0 4px;
        }
        .entity .info-btn:hover { color: #2196F3; }
        
        .entity .remove {
            color: #999; cursor: pointer; font-weight: bold;
            margin-left: 4px;
        }
        .entity .remove:hover { color: #f44336; }
        
        .noise-entity {
            background: #f5f5f5; border-color: #ddd;
        }
        
        .noise-entity:hover { background: #eeeeee; }
        
        .drop-zone {
            border: 2px dashed #ccc; border-radius: 8px;
            padding: 20px; text-align: center; color: #999;
            margin-top: 15px; transition: all 0.2s;
        }
        
        .drop-zone.drag-over {
            border-color: #4CAF50; background: #e8f5e9; color: #4CAF50;
        }
        
        .modal-overlay {
            display: none; position: fixed; top: 0; left: 0;
            width: 100%; height: 100%; background: rgba(0,0,0,0.5);
            justify-content: center; align-items: center; z-index: 1000;
        }
        
        .modal-overlay.active { display: flex; }
        
        .modal {
            background: white; padding: 25px; border-radius: 10px;
            max-width: 700px; width: 90%; max-height: 80vh; overflow-y: auto;
        }
        
        .modal h2 { margin-top: 0; color: #333; }
        
        .modal-actions {
            display: flex; gap: 10px; justify-content: flex-end; margin-top: 20px;
        }
        
        .context-list {
            max-height: 400px; overflow-y: auto; margin: 15px 0;
        }
        
        .context-item {
            background: #f8f9fa; padding: 12px; margin: 8px 0;
            border-radius: 6px; border-left: 3px solid #4CAF50;
            font-size: 13px; line-height: 1.5;
        }
        
        .context-source {
            font-size: 11px; color: #666; margin-bottom: 6px;
            padding-bottom: 6px; border-bottom: 1px solid #e0e0e0;
            font-weight: 500;
        }
        
        .context-source .work-name {
            color: #1976D2; font-weight: 600;
        }
        
        .context-source .doc-id {
            color: #888; margin-left: 8px;
        }
        
        .context-text {
            font-family: 'Courier New', monospace;
        }
        
        .context-item .highlight {
            background: #fff59d; padding: 1px 3px; border-radius: 2px;
            font-weight: bold;
        }
        
        .works-summary {
            margin: 10px 0; padding: 10px; background: #e3f2fd;
            border-radius: 6px; font-size: 13px;
        }
        
        .works-summary strong { color: #1565c0; }
        
        .entity-meta {
            display: grid; grid-template-columns: auto 1fr; gap: 8px 15px;
            margin-bottom: 15px; font-size: 14px;
        }
        
        .entity-meta dt { font-weight: 600; color: #666; }
        .entity-meta dd { margin: 0; }
        
        .variants-list {
            display: flex; flex-wrap: wrap; gap: 5px;
        }
        
        .variant-tag {
            background: #e3f2fd; padding: 3px 8px; border-radius: 12px;
            font-size: 12px;
        }
        
        .toast {
            position: fixed; bottom: 20px; right: 20px;
            background: #333; color: white; padding: 12px 24px;
            border-radius: 8px; opacity: 0; transition: opacity 0.3s;
            z-index: 1001;
        }
        
        .toast.show { opacity: 1; }
        
        .hidden { display: none !important; }
        
        .undo-stack {
            position: fixed; bottom: 20px; left: 20px;
            background: white; padding: 10px 15px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2); font-size: 13px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔧 """ + entity_type + """ Cluster Editor</h1>
        <div class="toolbar">
            <button class="btn btn-secondary" onclick="undo()">↶ Undo</button>
            <button class="btn btn-warning" onclick="mergeSelected()">Merge Selected</button>
            <button class="btn btn-primary" onclick="createNewCluster()">+ New Cluster</button>
            <button class="btn btn-primary" onclick="exportClusters()">💾 Export JSON</button>
        </div>
    </div>
    
    <div class="stats" id="stats"></div>
    
    <div class="search-container">
        <input type="text" class="search-box" id="search" 
               placeholder="Search entities..." onkeyup="filterEntities()">
    </div>
    
    <div class="main-container">
        <div class="clusters-panel" id="clusters-panel"></div>
        
        <div class="noise-panel">
            <h3>🔇 Unclustered (Noise)</h3>
            <div class="entity-list" id="noise-list"></div>
            <div class="drop-zone" id="noise-drop">
                Drop here to remove from cluster
            </div>
        </div>
    </div>
    
    <div class="modal-overlay" id="context-modal">
        <div class="modal">
            <h2 id="modal-title">Entity Details</h2>
            <dl class="entity-meta" id="entity-meta"></dl>
            <h3>Contexts (<span id="context-count">0</span>)</h3>
            <div class="context-list" id="context-list"></div>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal('context-modal')">Close</button>
            </div>
        </div>
    </div>
    
    <div class="modal-overlay" id="merge-modal">
        <div class="modal">
            <h2>Merge Clusters</h2>
            <p>Select clusters to merge by clicking the "Select" button on each cluster, then click "Merge Selected".</p>
            <div class="modal-actions">
                <button class="btn btn-secondary" onclick="closeModal('merge-modal')">Close</button>
            </div>
        </div>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <div class="undo-stack" id="undo-info">
        Undo stack: <span id="undo-count">0</span> actions
    </div>

    <script>
        // Initial data from Python
        let data = """ + json.dumps(initial_data, ensure_ascii=False) + """;
        
        let selectedClusters = new Set();
        let undoStack = [];
        let draggedEntity = null;
        let dragSource = null;
        
        // Calculate total occurrences (sum of all frequencies)
        function getTotalOccurrences() {
            let total = 0;
            data.clusters.forEach(c => {
                c.entities.forEach(e => total += e.frequency || 1);
            });
            data.noise.forEach(e => total += e.frequency || 1);
            return total;
        }
        
        function saveState() {
            undoStack.push(JSON.stringify(data));
            if (undoStack.length > 50) undoStack.shift();
            document.getElementById('undo-count').textContent = undoStack.length;
        }
        
        function undo() {
            if (undoStack.length === 0) {
                showToast('Nothing to undo');
                return;
            }
            data = JSON.parse(undoStack.pop());
            document.getElementById('undo-count').textContent = undoStack.length;
            render();
            showToast('Undone');
        }
        
        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 2000);
        }
        
        function updateStats() {
            const totalClustered = data.clusters.reduce((sum, c) => sum + c.entities.length, 0);
            const totalOccurrences = getTotalOccurrences();
            const uniqueEntities = totalClustered + data.noise.length;
            const html = `
                <span class="stat"><b>${data.clusters.length}</b> clusters</span>
                <span class="stat"><b>${totalClustered}</b> clustered unique entities</span>
                <span class="stat"><b>${data.noise.length}</b> noise</span>
                <span class="stat"><b>${uniqueEntities}</b> total unique</span>
                <span class="stat"><b>${totalOccurrences}</b> total occurrences</span>
                <span class="stat"><b>${selectedClusters.size}</b> selected for merge</span>
            `;
            document.getElementById('stats').innerHTML = html;
        }
        
        function render() {
            renderClusters();
            renderNoise();
            updateStats();
        }
        
        function renderClusters() {
            const panel = document.getElementById('clusters-panel');
            
            // Sort clusters by size (number of unique entities)
            const sorted = [...data.clusters].sort((a, b) => b.entities.length - a.entities.length);
            
            panel.innerHTML = sorted.map(cluster => {
                const totalFreq = cluster.entities.reduce((sum, e) => sum + (e.frequency || 1), 0);
                return `
                <div class="cluster-card ${selectedClusters.has(cluster.id) ? 'selected' : ''}" 
                     data-cluster-id="${cluster.id}"
                     ondragover="handleDragOver(event)" 
                     ondragleave="handleDragLeave(event)"
                     ondrop="handleDrop(event, ${cluster.id})">
                    <div class="cluster-header">
                        <span class="cluster-title">Cluster ${cluster.id} (${cluster.entities.length} unique, ${totalFreq} occurrences)</span>
                        <div class="cluster-actions">
                            <button class="btn-select ${selectedClusters.has(cluster.id) ? 'active' : ''}" 
                                    onclick="toggleSelect(${cluster.id})">
                                ${selectedClusters.has(cluster.id) ? '✓ Selected' : 'Select'}
                            </button>
                            <button class="btn-delete" onclick="deleteCluster(${cluster.id})">Delete</button>
                        </div>
                    </div>
                    <div class="entity-list">
                        ${cluster.entities.map(e => renderEntity(e, cluster.id)).join('')}
                    </div>
                </div>
            `}).join('');
        }
        
        function renderEntity(entity, clusterId) {
            const freq = entity.frequency || 1;
            let freqClass = '';
            if (freq >= 10) freqClass = 'very-high';
            else if (freq >= 5) freqClass = 'high';
            
            const freqBadge = `<span class="freq ${freqClass}" title="${freq} occurrences">×${freq}</span>`;
            const hasContexts = entity.contexts && entity.contexts.length > 0;
            const infoBtn = hasContexts ? 
                `<span class="info-btn" onclick="event.stopPropagation(); showEntityDetails(${entity.id}, ${clusterId})" title="View contexts">ℹ</span>` : '';
            
            // Show number of unique works/sources
            const numWorks = entity.works ? entity.works.length : 0;
            const worksIndicator = numWorks > 1 ? 
                `<span class="works-badge" title="${numWorks} different sources">📖${numWorks}</span>` : '';
            
            return `
                <div class="entity" 
                     draggable="true"
                     data-entity-id="${entity.id}"
                     data-cluster-id="${clusterId}"
                     ondragstart="handleDragStart(event, ${entity.id}, ${clusterId})"
                     ondragend="handleDragEnd(event)"
                     ondblclick="showEntityDetails(${entity.id}, ${clusterId})">
                    ${entity.text}
                    ${freqBadge}
                    ${worksIndicator}
                    ${infoBtn}
                    <span class="remove" onclick="event.stopPropagation(); moveToNoise(${entity.id}, ${clusterId})">×</span>
                </div>
            `;
        }
        
        function renderNoise() {
            const list = document.getElementById('noise-list');
            list.innerHTML = data.noise.map(e => {
                const freq = e.frequency || 1;
                let freqClass = '';
                if (freq >= 10) freqClass = 'very-high';
                else if (freq >= 5) freqClass = 'high';
                
                const hasContexts = e.contexts && e.contexts.length > 0;
                const infoBtn = hasContexts ? 
                    `<span class="info-btn" onclick="event.stopPropagation(); showEntityDetails(${e.id}, -1)" title="View contexts">ℹ</span>` : '';
                
                return `
                <div class="entity noise-entity" 
                     draggable="true"
                     data-entity-id="${e.id}"
                     data-cluster-id="-1"
                     ondragstart="handleDragStart(event, ${e.id}, -1)"
                     ondragend="handleDragEnd(event)"
                     ondblclick="showEntityDetails(${e.id}, -1)">
                    ${e.text}
                    <span class="freq ${freqClass}" title="${freq} occurrences">×${freq}</span>
                    ${infoBtn}
                </div>
            `}).join('');
            
            // Setup noise drop zone
            const dropZone = document.getElementById('noise-drop');
            dropZone.ondragover = handleDragOver;
            dropZone.ondragleave = handleDragLeave;
            dropZone.ondrop = (e) => handleDrop(e, -1);
        }
        
        function showEntityDetails(entityId, clusterId) {
            let entity;
            if (clusterId === -1) {
                entity = data.noise.find(e => e.id === entityId);
            } else {
                const cluster = data.clusters.find(c => c.id === clusterId);
                if (cluster) {
                    entity = cluster.entities.find(e => e.id === entityId);
                }
            }
            
            if (!entity) return;
            
            // Update modal title
            document.getElementById('modal-title').textContent = entity.text;
            
            // Update metadata
            const variants = entity.variants || [entity.text];
            const variantsHtml = variants.map(v => `<span class="variant-tag">${v}</span>`).join('');
            
            // Get unique works
            const works = entity.works || [];
            const worksHtml = works.length > 0 ? 
                `<div class="works-summary"><strong>${works.length}</strong> source(s): ${works.join(', ')}</div>` : '';
            
            document.getElementById('entity-meta').innerHTML = `
                <dt>Frequency:</dt>
                <dd><strong>${entity.frequency || 1}</strong> occurrences across texts</dd>
                <dt>Variants:</dt>
                <dd><div class="variants-list">${variantsHtml}</div></dd>
                <dt>Cluster:</dt>
                <dd>${clusterId >= 0 ? 'Cluster ' + clusterId : 'Noise (unclustered)'}</dd>
            `;
            
            // Update contexts with source information
            const contexts = entity.contexts || [];
            document.getElementById('context-count').textContent = contexts.length;
            
            let contextsHtml = '';
            if (contexts.length > 0) {
                contextsHtml = worksHtml + contexts.map(ctx => {
                    // Handle both old format (string) and new format (object with text/work/doc_id)
                    const ctxText = typeof ctx === 'string' ? ctx : (ctx.text || '');
                    const work = typeof ctx === 'object' ? (ctx.work || 'Unknown') : 'Unknown';
                    const docId = typeof ctx === 'object' ? ctx.doc_id : null;
                    
                    // Highlight the entity in context
                    const highlighted = ctxText.replace(
                        new RegExp(`(${entity.text})`, 'gi'),
                        '<span class="highlight">$1</span>'
                    );
                    
                    const docIdHtml = docId ? `<span class="doc-id">(doc #${docId})</span>` : '';
                    
                    return `
                        <div class="context-item">
                            <div class="context-source">
                                📖 <span class="work-name">${work}</span>${docIdHtml}
                            </div>
                            <div class="context-text">${highlighted}</div>
                        </div>
                    `;
                }).join('');
            } else {
                contextsHtml = '<p style="color: #999;">No context information available.</p>';
            }
            
            document.getElementById('context-list').innerHTML = contextsHtml;
            
            // Show modal
            document.getElementById('context-modal').classList.add('active');
        }
        
        function handleDragStart(event, entityId, clusterId) {
            draggedEntity = entityId;
            dragSource = clusterId;
            event.target.classList.add('dragging');
            event.dataTransfer.effectAllowed = 'move';
        }
        
        function handleDragEnd(event) {
            event.target.classList.remove('dragging');
            document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
        }
        
        function handleDragOver(event) {
            event.preventDefault();
            event.currentTarget.classList.add('drag-over');
        }
        
        function handleDragLeave(event) {
            event.currentTarget.classList.remove('drag-over');
        }
        
        function handleDrop(event, targetClusterId) {
            event.preventDefault();
            event.currentTarget.classList.remove('drag-over');
            
            if (draggedEntity === null || dragSource === targetClusterId) return;
            
            saveState();
            moveEntity(draggedEntity, dragSource, targetClusterId);
            draggedEntity = null;
            dragSource = null;
        }
        
        function moveEntity(entityId, fromCluster, toCluster) {
            let entity;
            
            // Remove from source
            if (fromCluster === -1) {
                const idx = data.noise.findIndex(e => e.id === entityId);
                if (idx >= 0) {
                    entity = data.noise.splice(idx, 1)[0];
                }
            } else {
                const cluster = data.clusters.find(c => c.id === fromCluster);
                if (cluster) {
                    const idx = cluster.entities.findIndex(e => e.id === entityId);
                    if (idx >= 0) {
                        entity = cluster.entities.splice(idx, 1)[0];
                    }
                }
            }
            
            if (!entity) return;
            
            // Add to target
            if (toCluster === -1) {
                data.noise.push(entity);
            } else {
                const cluster = data.clusters.find(c => c.id === toCluster);
                if (cluster) {
                    cluster.entities.push(entity);
                }
            }
            
            // Remove empty clusters
            data.clusters = data.clusters.filter(c => c.entities.length > 0);
            
            render();
            showToast('Entity moved');
        }
        
        function moveToNoise(entityId, clusterId) {
            saveState();
            moveEntity(entityId, clusterId, -1);
        }
        
        function toggleSelect(clusterId) {
            if (selectedClusters.has(clusterId)) {
                selectedClusters.delete(clusterId);
            } else {
                selectedClusters.add(clusterId);
            }
            render();
        }
        
        function mergeSelected() {
            if (selectedClusters.size < 2) {
                showToast('Select at least 2 clusters to merge');
                return;
            }
            
            saveState();
            
            const clusterIds = Array.from(selectedClusters);
            const targetId = Math.min(...clusterIds);
            const targetCluster = data.clusters.find(c => c.id === targetId);
            
            // Merge all into target
            clusterIds.forEach(id => {
                if (id !== targetId) {
                    const source = data.clusters.find(c => c.id === id);
                    if (source) {
                        targetCluster.entities.push(...source.entities);
                    }
                }
            });
            
            // Remove merged clusters
            data.clusters = data.clusters.filter(c => !clusterIds.includes(c.id) || c.id === targetId);
            
            selectedClusters.clear();
            render();
            showToast(`Merged ${clusterIds.length} clusters`);
        }
        
        function createNewCluster() {
            saveState();
            
            const newId = data.nextClusterId++;
            data.clusters.push({
                id: newId,
                entities: []
            });
            
            render();
            showToast(`Created cluster ${newId} - drag entities to it`);
        }
        
        function deleteCluster(clusterId) {
            const cluster = data.clusters.find(c => c.id === clusterId);
            if (!cluster) return;
            
            if (cluster.entities.length > 0) {
                if (!confirm(`Move ${cluster.entities.length} entities to noise?`)) return;
            }
            
            saveState();
            
            // Move entities to noise
            data.noise.push(...cluster.entities);
            
            // Remove cluster
            data.clusters = data.clusters.filter(c => c.id !== clusterId);
            selectedClusters.delete(clusterId);
            
            render();
            showToast('Cluster deleted');
        }
        
        function filterEntities() {
            const search = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.entity').forEach(el => {
                const text = el.textContent.toLowerCase();
                el.classList.toggle('hidden', !text.includes(search));
            });
            document.querySelectorAll('.cluster-card').forEach(card => {
                const visibleEntities = card.querySelectorAll('.entity:not(.hidden)');
                card.classList.toggle('hidden', visibleEntities.length === 0 && search !== '');
            });
        }
        
        function closeModal(id) {
            document.getElementById(id).classList.remove('active');
        }
        
        function exportClusters() {
            const exportData = {
                entityType: data.entityType,
                exportedAt: new Date().toISOString(),
                clusters: data.clusters.map(c => ({
                    id: c.id,
                    entities: c.entities.map(e => ({
                        text: e.text,
                        frequency: e.frequency || 1,
                        variants: e.variants || [e.text],
                        works: e.works || [],
                        contexts: (e.contexts || []).map(ctx => ({
                            text: typeof ctx === 'string' ? ctx : ctx.text,
                            work: typeof ctx === 'object' ? ctx.work : 'Unknown',
                            doc_id: typeof ctx === 'object' ? ctx.doc_id : null
                        }))
                    }))
                })),
                noise: data.noise.map(e => ({
                    text: e.text,
                    frequency: e.frequency || 1,
                    variants: e.variants || [e.text],
                    works: e.works || [],
                    contexts: (e.contexts || []).map(ctx => ({
                        text: typeof ctx === 'string' ? ctx : ctx.text,
                        work: typeof ctx === 'object' ? ctx.work : 'Unknown',
                        doc_id: typeof ctx === 'object' ? ctx.doc_id : null
                    }))
                })),
                statistics: {
                    totalClusters: data.clusters.length,
                    totalClusteredUnique: data.clusters.reduce((sum, c) => sum + c.entities.length, 0),
                    totalClusteredOccurrences: data.clusters.reduce((sum, c) => 
                        sum + c.entities.reduce((s, e) => s + (e.frequency || 1), 0), 0),
                    totalNoise: data.noise.length,
                    totalOccurrences: getTotalOccurrences()
                }
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${data.entityType}_clusters_curated.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            showToast('Exported to JSON');
        }
        
        // Close modal on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
            }
        });
        
        // Close modal on background click
        document.querySelectorAll('.modal-overlay').forEach(overlay => {
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) overlay.classList.remove('active');
            });
        });
        
        // Initial render
        render();
    </script>
</body>
</html>"""
    
    output_path = Path(output_dir) / f"{entity_type.lower()}_cluster_editor.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("Entity Clustering v3 - Interactive Editor")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  String weight: {STRING_WEIGHT} ({STRING_WEIGHT*100:.0f}%)")
    print(f"  Embedding weight: {EMBEDDING_WEIGHT} ({EMBEDDING_WEIGHT*100:.0f}%)")
    print()
    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load entities
    print(f"Loading entities from {ENTITIES_FILE}...")
    entities_by_type = load_entities(ENTITIES_FILE)
    
    total = sum(len(v) for v in entities_by_type.values())
    print(f"Loaded {total} entities")
    
    for etype, entities in entities_by_type.items():
        print(f"\n{'='*70}")
        print(f"Processing {etype}")
        print("="*70)
        
        entities = deduplicate_entities(entities)
        print(f"After deduplication: {len(entities)}")
        
        if len(entities) < MIN_CLUSTER_SIZE:
            print("Skipping - too few entities")
            continue
        
        texts = [e["text"] for e in entities]
        embeddings = get_embeddings(texts, MODEL_PATH)
        
        hybrid_dist, string_dist = compute_hybrid_distance_matrix(entities, embeddings)
        
        print("Clustering...")
        labels = cluster_entities(hybrid_dist)
        
        n_clusters = len(set(l for l in labels if l >= 0))
        n_noise = sum(1 for l in labels if l < 0)
        print(f"Found {n_clusters} clusters, {n_noise} noise")
        
        # Create interactive editor
        editor_path = create_interactive_editor(entities, labels, etype, output_dir)
        print(f"Created editor: {editor_path}")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"\nOutput: {OUTPUT_DIR}")
    print("Open the *_cluster_editor.html files in a browser to curate clusters.")
    print("\nFeatures:")
    print("  - Drag entities between clusters")
    print("  - Select multiple clusters and merge them")
    print("  - Create new empty clusters")
    print("  - Move entities to noise (remove from cluster)")
    print("  - Export curated clusters to JSON")
    print("  - Undo support")

if __name__ == "__main__":
    main()