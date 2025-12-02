# ============================================================================
# 4. VISUALIZATION & EXPORT (DEBUG MODE)
# ============================================================================

def visualize_and_export(parsed_data, nlp, stats, best_config):
    print("\n" + "="*80)
    print("STEP 4: VISUAL INSPECTION (DEBUG MODE)")
    print("="*80)
    print("Legend:")
    print("  🔹 EXPECTED:  The exact string from your JSON label")
    print("  🔸 EXTRACTED: The actual tokens spaCy captured (or FAILED)")
    print("  ⬜ TOKENS:    Shows how spaCy split the text (look for [   ] spaces!)")
    print("="*80)

    # 1. BUCKET THE DATA
    # We want to prioritize showing errors and rescued items
    interesting_samples = []
    
    for text, entities in parsed_data:
        if not entities: continue
        
        doc = nlp.make_doc(text)
        doc_status = "PERFECT" # Default
        
        entry_details = []
        
        for ent in entities:
            # Check Alignment
            span = doc.char_span(ent['start'], ent['end'], label=ent['label'])
            
            if span:
                status = "✅ OK"
                extracted_text = span.text
                token_count = len(span)
            else:
                # Try Rescue (Expand)
                span_expand = doc.char_span(ent['start'], ent['end'], label=ent['label'], alignment_mode="expand")
                if span_expand:
                    status = "⚠️ RESCUED"
                    doc_status = "HAS_RESCUED" if doc_status != "HAS_ERROR" else doc_status
                    extracted_text = span_expand.text
                    token_count = len(span_expand)
                else:
                    status = "❌ MISALIGNED"
                    doc_status = "HAS_ERROR"
                    extracted_text = "NONE (Label cuts through a token)"
                    token_count = 0
            
            entry_details.append({
                'label': ent['label'],
                'expected': ent['text'],
                'extracted': extracted_text,
                'status': status,
                'tokens': token_count
            })
            
        if doc_status in ["HAS_ERROR", "HAS_RESCUED"]:
            interesting_samples.append({'text': text, 'doc': doc, 'details': entry_details, 'type': doc_status})
    
    # Add a few perfect ones for comparison
    perfect_samples = []
    for text, entities in parsed_data:
        if len(perfect_samples) >= 2: break
        # Simple check if this doc was already caught as an error
        if not any(s['text'] == text for s in interesting_samples):
             doc = nlp.make_doc(text)
             # Reprocess briefly to match structure
             details = []
             for ent in entities:
                 span = doc.char_span(ent['start'], ent['end'])
                 if span: details.append({'label': ent['label'], 'expected': ent['text'], 'extracted': span.text, 'status': "✅ OK", 'tokens': len(span)})
             perfect_samples.append({'text': text, 'doc': doc, 'details': details, 'type': "PERFECT"})

    # Combine: Errors first, then Rescued, then 2 Perfect examples
    final_display = sorted(interesting_samples, key=lambda x: x['type'] == 'HAS_ERROR', reverse=True)[:10] + perfect_samples

    # 2. PRINT THE SAMPLES
    for i, sample in enumerate(final_display, 1):
        print(f"\n📄 DOC {i} [{sample['type']}]")
        print("-" * 80)
        print(f"TEXT:   {sample['text']}")
        
        # VISUALIZE TOKENS (Crucial for seeing spaces)
        # We put brackets [ ] around tokens so you can see invisible spaces
        token_view = " ".join([f"[{t.text}]" for t in sample['doc']])
        print(f"TOKENS: {token_view}")
        print("-" * 80)
        
        print(f"{'STATUS':<15} {'LABEL':<15} {'EXPECTED (JSON)':<20} {'EXTRACTED (SPACY)':<25} {'TOKENS'}")
        print("-" * 80)
        
        for d in sample['details']:
            # Truncate for display
            exp = (d['expected'][:18] + '..') if len(d['expected']) > 18 else d['expected']
            ext = (d['extracted'][:23] + '..') if len(d['extracted']) > 23 else d['extracted']
            
            print(f"{d['status']:<15} {d['label']:<15} {exp:<20} {ext:<25} {d['tokens']}")

    # 3. EXPORT REPORT
    print("\n" + "="*80)
    print("🚀 SAVING CONFIG")
    print("="*80)
    
    if IS_SPANCAT and best_config:
        print(f"Recommended Suggester: {best_config['name']}")
        print(f"Config: min_size={min(best_config['sizes'])}, max_size={max(best_config['sizes'])}")
        
        report_text = f"""
        SUGGESTED CONFIGURATION:
        [components.spancat.suggester]
        @misc = "spacy.ngram_range_suggester.v1"
        min_size = {min(best_config['sizes'])}
        max_size = {max(best_config['sizes'])}
        """
        
        with open("/dbfs/tmp/spacy_eda_report.txt", "w") as f:
            f.write(report_text)
        print("✅ Config saved to /dbfs/tmp/spacy_eda_report.txt")



"""
ULTIMATE SPACY EDA & CONFIG VALIDATOR
-------------------------------------
1. Analyzes Data Quality (Misalignment, Whitespace, Duplicates)
2. Calculates Token Statistics
3. VALIDATES Suggester Coverage (The "Recall Ceiling")
4. Generates Production Config
"""

import pandas as pd
import json
import spacy
from collections import Counter, defaultdict
from spacy.tokens import DocBin, Span
import numpy as np
from typing import Dict, List, Tuple, Set
import warnings
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Databricks Table Name
TRAIN_TABLE = "trainingforengdata_redacted_relabelled"
TEXT_COLUMN = "TEXT"
LABEL_COLUMN = "ENTITIES_LABEL"
SPANCAT_KEY = "sc"

# Model Config
BASE_MODEL = "en_core_web_sm"
IS_SPANCAT = True  # Set False if using standard NER (Transition-based)
SAMPLE_SIZE = None # Set to integer (e.g., 50000) for testing, None for full run

# ============================================================================
# 1. SETUP & LOADING
# ============================================================================

def load_and_parse():
    print(f"\nLOADING SPACY MODEL: {BASE_MODEL}")
    try:
        nlp = spacy.load(BASE_MODEL)
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", BASE_MODEL])
        nlp = spacy.load(BASE_MODEL)

    print(f"LOADING DATA: {TRAIN_TABLE}")
    # Spark session is implicit in Databricks
    try:
        df = spark.table(TRAIN_TABLE).toPandas()
    except NameError:
        # Fallback for local testing if spark isn't available
        print("⚠️ Spark not detected. Expecting local dataframe or mock.")
        return [], nlp, None 

    if SAMPLE_SIZE:
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
        print(f"⚠️ Sampling {len(df)} records for analysis")

    print(f"Total Records: {len(df):,}")
    
    # Parse JSON
    print("PARSING ANNOTATIONS...")
    parsed_data = []
    parse_errors = 0
    
    for idx, row in df.iterrows():
        text = str(row[TEXT_COLUMN])
        label_info = row[LABEL_COLUMN]
        
        if pd.isna(label_info) or not str(label_info).strip():
            parsed_data.append((text, []))
            continue

        try:
            annotations = json.loads(label_info)
            entities = []
            for anno in annotations:
                if all(k in anno for k in ["start_char", "end_char", "label"]):
                    entities.append({
                        'start': anno["start_char"],
                        'end': anno["end_char"],
                        'label': anno["label"],
                        'text': text[anno["start_char"]:anno["end_char"]]
                    })
            parsed_data.append((text, entities))
        except json.JSONDecodeError:
            parse_errors += 1
            parsed_data.append((text, []))

    print(f"✅ Parsed {len(parsed_data):,} docs. (Errors: {parse_errors})")
    return parsed_data, nlp, df

# ============================================================================
# 2. HIGH-PERFORMANCE TOKENIZATION ANALYSIS
# ============================================================================

def analyze_tokenization_batch(parsed_data, nlp):
    """
    Optimized analysis using nlp.pipe for speed.
    """
    print("\n" + "="*80)
    print("STEP 2: BATCH TOKENIZATION & QUALITY CHECK")
    print("="*80)

    stats = {
        'span_lens': [],
        'doc_lens': [],
        'misaligned': [],
        'whitespace_issues': [], # " Mastercard " vs "Mastercard"
        'newline_issues': [],    # Entities spanning \n
        'label_counts': Counter()
    }

    # Generator for pipe
    texts = (t for t, _ in parsed_data)
    
    # PIPELINE OPTIMIZATION: Disable unused components for 10x speed
    print("Running spaCy pipeline (tokenizer only)...")
    docs = nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

    for (text, entities), doc in zip(parsed_data, docs):
        stats['doc_lens'].append(len(doc))
        
        for ent in entities:
            stats['label_counts'][ent['label']] += 1
            
            # 1. Whitespace Check
            if ent['text'].strip() != ent['text']:
                stats['whitespace_issues'].append(ent['text'])

            # 2. Newline Check
            if '\n' in ent['text']:
                stats['newline_issues'].append(ent['text'])

            # 3. Alignment & Token Length
            span = doc.char_span(ent['start'], ent['end'], label=ent['label'])
            
            if span is None:
                # Try contract mode to see if it's recoverable
                contract_span = doc.char_span(ent['start'], ent['end'], alignment_mode="contract")
                recoverable = "YES" if contract_span else "NO"
                
                stats['misaligned'].append({
                    'text': ent['text'],
                    'context': text[:50],
                    'recoverable': recoverable
                })
            else:
                stats['span_lens'].append(len(span))

    # --- REPORTING ---
    print(f"\n📊 TOKEN STATISTICS:")
    if stats['span_lens']:
        print(f"  Min Span Tokens: {min(stats['span_lens'])}")
        print(f"  Max Span Tokens: {max(stats['span_lens'])}")
        print(f"  Avg Span Tokens: {np.mean(stats['span_lens']):.2f}")
        print(f"  99th Percentile: {np.percentile(stats['span_lens'], 99):.1f}")
    
    print(f"\n🗑️ DIRTY DATA DETECTION:")
    print(f"  Misaligned Spans:      {len(stats['misaligned']):<5} (Can't map char indices to tokens)")
    print(f"  Whitespace Issues:     {len(stats['whitespace_issues']):<5} (Labels contain leading/trailing space)")
    print(f"  Newline Spans:         {len(stats['newline_issues']):<5} (Entities crossing line breaks)")

    if stats['whitespace_issues']:
        print(f"  → Example Whitespace: '{stats['whitespace_issues'][0]}'")
    
    return stats

# ============================================================================
# 3. THE RECALL CEILING (Suggester Validation)
# ============================================================================

def validate_suggester_coverage(parsed_data, nlp, stats):
    """
    Runs the ACTUAL suggester function to calculate Theoretical Max Recall.
    """
    if not IS_SPANCAT:
        print("\nSkipping Suggester Test (Not using Spancat)")
        return {}
        
    print("\n" + "="*80)
    print("STEP 3: SUGGESTER VALIDATION (RECALL CEILING)")
    print("="*80)

    try:
        from spacy.pipeline.spancat import ngram_suggester
    except ImportError:
        print("❌ Requires spaCy v3.1+ for spancat components")
        return {}

    if not stats['span_lens']:
        print("No valid spans to test.")
        return {}

    # Define Candidates
    min_t = min(stats['span_lens'])
    max_t = max(stats['span_lens'])
    p95_t = int(np.percentile(stats['span_lens'], 95))
    
    configs = {
        'Conservative (p95)': list(range(min_t, p95_t + 1)),
        'Balanced (Max)':     list(range(min_t, max_t + 1)),
        'Default (1-3)':      [1, 2, 3]
    }

    # Test on Sample (Slow operation)
    test_size = 200
    print(f"Testing on random {test_size} documents...")
    import random
    test_batch = random.sample([d for d in parsed_data if d[1]], min(test_size, len(parsed_data)))

    best_config = None
    best_score = -1

    print(f"\n{'CONFIG NAME':<20} {'SIZES':<10} {'COVERAGE (MAX RECALL)':<25} {'CANDIDATES/DOC'}")
    print("-" * 80)

    for name, sizes in configs.items():
        gold_spans = 0
        covered_spans = 0
        total_candidates = 0

        for text, entities in test_batch:
            doc = nlp.make_doc(text)
            
            # 1. Run Suggester
            candidates = ngram_suggester([doc], sizes=sizes)
            cand_set = set()
            if candidates.lengths[0] > 0:
                for i in range(candidates.lengths[0]):
                    s, e = candidates.data[i]
                    cand_set.add((int(s), int(e)))
            
            total_candidates += len(cand_set)
            
            # 2. Check Gold Coverage
            for ent in entities:
                span = doc.char_span(ent['start'], ent['end'])
                if span:
                    gold_spans += 1
                    if (span.start, span.end) in cand_set:
                        covered_spans += 1
        
        coverage = (covered_spans / gold_spans * 100) if gold_spans else 0
        avg_cand = total_candidates / len(test_batch)
        
        # Scoring logic for recommendation (Penalty for high candidate count)
        score = coverage
        if avg_cand > 500: score -= 10 
        if score > best_score:
            best_score = score
            best_config = {'name': name, 'sizes': sizes, 'coverage': coverage, 'avg': avg_cand}

        status = "✅" if coverage == 100 else "⚠️"
        print(f"{name:<20} {str(min(sizes))+'-'+str(max(sizes)):<10} {status} {coverage:.2f}%{' '*15} {avg_cand:.1f}")

    return best_config

# ============================================================================
# 4. VISUALIZATION & EXPORT
# ============================================================================

def visualize_and_export(parsed_data, nlp, stats, best_config):
    print("\n" + "="*80)
    print("STEP 4: VISUAL INSPECTION")
    print("="*80)

    # Show 2 complex examples
    samples = [d for d in parsed_data if len(d[1]) > 1][:2]
    
    for text, entities in samples:
        doc = nlp.make_doc(text)
        print(f"\nTEXT: {text[:100]}...")
        for ent in entities:
            span = doc.char_span(ent['start'], ent['end'])
            token_status = f"Tokens: {len(span)}" if span else "❌ MISALIGNED"
            print(f"  - [{ent['label']}] '{ent['text']}' -> {token_status}")

    # Generate Report
    print("\n" + "="*80)
    print("🚀 FINAL RECOMMENDATION")
    print("="*80)

    report = []
    report.append("DATA QUALITY REPORT")
    report.append(f"Misaligned Entities: {len(stats['misaligned'])}")
    report.append(f"Whitespace Errors:   {len(stats['whitespace_issues'])}")
    
    if IS_SPANCAT and best_config:
        rec_str = (
            f"\nRecommended Suggester: {best_config['name']}\n"
            f"Sizes: {min(best_config['sizes'])}-{max(best_config['sizes'])}\n"
            f"Theoretical Max Recall: {best_config['coverage']:.2f}%"
        )
        report.append(rec_str)
        print(rec_str)
        
        print("\n📋 PASTE THIS INTO YOUR CONFIG.CFG:")
        print(f"[components.spancat.suggester]")
        print(f"@misc = \"spacy.ngram_range_suggester.v1\"")
        print(f"min_size = {min(best_config['sizes'])}")
        print(f"max_size = {max(best_config['sizes'])}")
    else:
        print("Standard NER (transition-based) selected. No suggester config needed.")

    # Save to DBFS
    with open("/dbfs/tmp/spacy_eda_report.txt", "w") as f:
        f.write("\n".join(report))
    print("\nReport saved to /dbfs/tmp/spacy_eda_report.txt")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    data, nlp, df = load_and_parse()
    if data:
        stats = analyze_tokenization_batch(data, nlp)
        best_cfg = validate_suggester_coverage(data, nlp, stats)
        visualize_and_export(data, nlp, stats, best_cfg)
