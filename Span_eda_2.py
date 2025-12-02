"""
PRODUCTION-GRADE SPACY NER EDA & FINE-TUNING ANALYZER
======================================================
Comprehensive analysis for SpanCat fine-tuning using REAL spaCy libraries:

1. Token-level statistics (GROUND TRUTH spans)
2. Suggester validation (ngram + SpanFinder comparison)
3. Span Distinctiveness (SD) calculation
4. Space token verification (CRITICAL for offset validation)
5. Grouped misalignment analysis
6. Per-label quality metrics
7. Fine-tuning recommendations
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
import re

# Try to import SpanFinder (optional)
try:
    from spacy_experimental.span_finder.span_finder_suggester import span_finder_suggester
    SPANFINDER_AVAILABLE = True
except ImportError:
    SPANFINDER_AVAILABLE = False
    print("⚠️ spacy-experimental not available. Install with: pip install spacy-experimental")
    print("   SpanFinder comparison will be skipped.")

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
SAMPLE_SIZE = None  # Set to integer (e.g., 50000) for testing, None for full run

# ============================================================================
# 1. SETUP & LOADING
# ============================================================================

def load_and_parse():
    """Load data from Delta table and parse JSON annotations"""
    print("="*80)
    print("STEP 1: DATA LOADING & PARSING")
    print("="*80)
    
    print(f"\nLOADING SPACY MODEL: {BASE_MODEL}")
    try:
        nlp = spacy.load(BASE_MODEL)
    except OSError:
        import subprocess
        subprocess.run([sys.executable, "-m", "spacy", "download", BASE_MODEL])
        nlp = spacy.load(BASE_MODEL)

    print(f"LOADING DATA: {TRAIN_TABLE}")
    try:
        df = spark.table(TRAIN_TABLE).toPandas()
    except NameError:
        print("⚠️ Spark not detected. Expecting local dataframe or mock.")
        return [], nlp, None 

    if SAMPLE_SIZE:
        df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
        print(f"⚠️ Sampling {len(df)} records for analysis")

    print(f"Total Records: {len(df):,}")
    
    # Parse JSON annotations
    print("PARSING JSON ANNOTATIONS...")
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

    print(f"✅ Parsed {len(parsed_data):,} documents")
    print(f"   Parse errors: {parse_errors}")
    
    return parsed_data, nlp, df

# ============================================================================
# HELPER FUNCTIONS FOR ADVANCED METRICS
# ============================================================================

def calculate_span_distinctiveness(parsed_data, nlp, sample_size=1000):
    """
    Calculate Span Distinctiveness (SD) score.
    SD measures how "distinctive" entity spans are compared to random text.
    SD > 1.0 = spans are distinctive (good)
    SD < 1.0 = spans look like regular text (hard task)
    
    Formula from spaCy: SD = (entity_token_freq) / (overall_token_freq)
    """
    print("\n" + "="*80)
    print("CALCULATING SPAN DISTINCTIVENESS (SD)")
    print("="*80)
    print("SD > 1.0 = Entity spans are distinctive (easier to learn)")
    print("SD < 1.0 = Entity spans look like regular text (harder)")
    print("="*80)
    
    # Sample for performance
    import random
    sample = random.sample(parsed_data, min(sample_size, len(parsed_data)))
    
    # Count token frequencies in entities vs. overall
    entity_token_counts = Counter()
    overall_token_counts = Counter()
    total_entity_tokens = 0
    total_overall_tokens = 0
    
    for text, entities in sample:
        doc = nlp.make_doc(text)
        
        # Overall token counts
        for token in doc:
            if not token.is_space:  # Ignore space tokens
                overall_token_counts[token.text.lower()] += 1
                total_overall_tokens += 1
        
        # Entity token counts
        for ent in entities:
            span = doc.char_span(ent['start'], ent['end'])
            if span:
                for token in span:
                    if not token.is_space:
                        entity_token_counts[token.text.lower()] += 1
                        total_entity_tokens += 1
    
    # Calculate SD score
    if total_entity_tokens == 0 or total_overall_tokens == 0:
        print("⚠️ Not enough data to calculate SD")
        return None
    
    # Average frequency in entities
    entity_avg_freq = sum(entity_token_counts.values()) / len(entity_token_counts) if entity_token_counts else 0
    # Average frequency overall
    overall_avg_freq = sum(overall_token_counts.values()) / len(overall_token_counts) if overall_token_counts else 0
    
    sd_score = entity_avg_freq / overall_avg_freq if overall_avg_freq > 0 else 0
    
    print(f"\n📊 SPAN DISTINCTIVENESS SCORE: {sd_score:.3f}")
    
    if sd_score > 1.5:
        verdict = "✅ EXCELLENT - Entities are very distinctive"
    elif sd_score > 1.0:
        verdict = "✓ GOOD - Entities are somewhat distinctive"
    elif sd_score > 0.8:
        verdict = "⚠️ MODERATE - Entities blend with regular text"
    else:
        verdict = "❌ POOR - Entities are not distinctive (hard task)"
    
    print(f"   Interpretation: {verdict}")
    print(f"\n   Entity token unique count: {len(entity_token_counts):,}")
    print(f"   Overall token unique count: {len(overall_token_counts):,}")
    
    # Show most distinctive entity tokens
    print(f"\n🔍 MOST DISTINCTIVE ENTITY TOKENS:")
    entity_freq = {k: entity_token_counts[k]/total_entity_tokens for k in entity_token_counts}
    overall_freq = {k: overall_token_counts[k]/total_overall_tokens for k in overall_token_counts}
    
    distinctiveness = {}
    for token in entity_token_counts:
        if token in overall_freq:
            distinctiveness[token] = entity_freq[token] / overall_freq[token]
    
    for token, score in sorted(distinctiveness.items(), key=lambda x: -x[1])[:10]:
        print(f"   '{token}': {score:.2f}x more common in entities")
    
    return sd_score

def detect_space_tokens(parsed_data, nlp, max_examples=20):
    """
    Detect transactions with SPACE TOKENS in tokenization.
    This is CRITICAL - if JSON offsets don't account for space tokens, alignment fails.
    """
    print("\n" + "="*80)
    print("SPACE TOKEN DETECTION (CRITICAL FOR ALIGNMENT)")
    print("="*80)
    print("Checking if tokenizer creates space tokens [ ] or [  ]...")
    print("If present, JSON offsets MUST account for them!")
    print("="*80)
    
    space_token_docs = []
    total_space_tokens = 0
    
    for text, entities in parsed_data[:1000]:  # Check first 1000
        doc = nlp.make_doc(text)
        
        has_space_token = False
        space_tokens_in_doc = []
        
        for i, token in enumerate(doc):
            if token.is_space or token.text.strip() == '':
                has_space_token = True
                total_space_tokens += 1
                space_tokens_in_doc.append({
                    'index': i,
                    'text': repr(token.text),  # Show with quotes
                    'start_char': token.idx,
                    'end_char': token.idx + len(token.text)
                })
        
        if has_space_token:
            space_token_docs.append({
                'text': text,
                'doc': doc,
                'space_tokens': space_tokens_in_doc,
                'entities': entities
            })
    
    print(f"\n📊 SPACE TOKEN STATISTICS:")
    print(f"   Documents checked: 1,000")
    print(f"   Documents with space tokens: {len(space_token_docs):,}")
    print(f"   Total space tokens found: {total_space_tokens:,}")
    
    if total_space_tokens > 0:
        print(f"\n⚠️ SPACE TOKENS DETECTED!")
        print(f"   → Your JSON offsets MUST account for these tokens")
        print(f"   → If offsets skip space tokens, alignment will FAIL")
        
        print(f"\n🔍 EXAMPLES OF SPACE TOKENS (showing {min(max_examples, len(space_token_docs))}):")
        
        for i, example in enumerate(space_token_docs[:max_examples], 1):
            print(f"\n   Example {i}:")
            print(f"   TEXT: {example['text'][:80]}...")
            print(f"   TOKENS: {[t.text for t in example['doc']]}")
            print(f"   Space tokens at indices: {[st['index'] for st in example['space_tokens']]}")
            
            # Check if entities are affected
            for ent in example['entities']:
                span = example['doc'].char_span(ent['start'], ent['end'])
                if span:
                    # Check if span contains space tokens
                    has_space_in_span = any(t.is_space for t in span)
                    if has_space_in_span:
                        print(f"      ⚠️ Entity '{ent['text']}' ({ent['label']}) contains SPACE TOKEN")
                        print(f"         Token span: {[t.text for t in span]}")
    else:
        print(f"\n✅ NO SPACE TOKENS DETECTED")
        print(f"   → Tokenizer does not create space tokens")
        print(f"   → Standard char offset validation applies")
    
    return space_token_docs

def verify_offset_alignment(parsed_data, nlp, sample_size=100):
    """
    Verify that JSON character offsets correctly align with token boundaries,
    including space tokens if present.
    """
    print("\n" + "="*80)
    print("OFFSET ALIGNMENT VERIFICATION")
    print("="*80)
    
    import random
    sample = random.sample([d for d in parsed_data if d[1]], 
                          min(sample_size, len([d for d in parsed_data if d[1]])))
    
    alignment_issues = {
        'cuts_through_token': [],  # Offset splits a token
        'missing_space_token': [],  # Offset skips a space token
        'boundary_mismatch': [],   # Start/end don't align with token boundaries
        'perfect': 0
    }
    
    for text, entities in sample:
        doc = nlp.make_doc(text)
        
        for ent in entities:
            span = doc.char_span(ent['start'], ent['end'])
            
            if span is None:
                # Try to diagnose WHY it failed
                # Find which token the start/end falls into
                start_token_idx = None
                end_token_idx = None
                
                for i, token in enumerate(doc):
                    if token.idx <= ent['start'] < token.idx + len(token.text):
                        start_token_idx = i
                    if token.idx < ent['end'] <= token.idx + len(token.text):
                        end_token_idx = i
                
                issue_type = 'cuts_through_token'
                if start_token_idx is not None and end_token_idx is not None:
                    # Check if we're missing a space token
                    expected_tokens = list(range(start_token_idx, end_token_idx + 1))
                    has_space = any(doc[i].is_space for i in expected_tokens if i < len(doc))
                    if has_space:
                        issue_type = 'missing_space_token'
                
                alignment_issues[issue_type].append({
                    'text': text,
                    'entity_text': ent['text'],
                    'label': ent['label'],
                    'start': ent['start'],
                    'end': ent['end'],
                    'tokens': [t.text for t in doc],
                    'start_token_idx': start_token_idx,
                    'end_token_idx': end_token_idx
                })
            else:
                # Check for boundary issues (start/end not at token boundaries)
                if span.start_char != ent['start'] or span.end_char != ent['end']:
                    alignment_issues['boundary_mismatch'].append({
                        'text': text,
                        'entity_text': ent['text'],
                        'expected_start': ent['start'],
                        'expected_end': ent['end'],
                        'actual_start': span.start_char,
                        'actual_end': span.end_char
                    })
                else:
                    alignment_issues['perfect'] += 1
    
    # Report
    total_checked = sum(len(ents) for _, ents in sample)
    print(f"\n📊 OFFSET VERIFICATION RESULTS ({total_checked:,} spans checked):")
    print(f"   ✅ Perfect alignment:     {alignment_issues['perfect']:,} "
          f"({alignment_issues['perfect']/total_checked*100:.2f}%)")
    print(f"   ❌ Cuts through token:    {len(alignment_issues['cuts_through_token']):,}")
    print(f"   ⚠️ Missing space token:   {len(alignment_issues['missing_space_token']):,}")
    print(f"   ⚠️ Boundary mismatch:     {len(alignment_issues['boundary_mismatch']):,}")
    
    # Show examples
    if alignment_issues['cuts_through_token']:
        print(f"\n🔍 EXAMPLE: Offset cuts through token:")
        ex = alignment_issues['cuts_through_token'][0]
        print(f"   Entity: '{ex['entity_text']}' (Label: {ex['label']})")
        print(f"   Offsets: [{ex['start']}:{ex['end']}]")
        print(f"   Tokens: {ex['tokens']}")
        print(f"   → Offset falls INSIDE a token boundary")
    
    if alignment_issues['missing_space_token']:
        print(f"\n🔍 EXAMPLE: Offset misses space token:")
        ex = alignment_issues['missing_space_token'][0]
        print(f"   Entity: '{ex['entity_text']}' (Label: {ex['label']})")
        print(f"   Offsets: [{ex['start']}:{ex['end']}]")
        print(f"   Tokens: {ex['tokens']}")
        print(f"   → Likely skipping a SPACE TOKEN in the middle")
    
    return alignment_issues

# ============================================================================
# 2. DETAILED TOKENIZATION ANALYSIS
# ============================================================================

def analyze_tokenization_comprehensive(parsed_data, nlp):
    """
    Comprehensive analysis of GROUND TRUTH annotations at token level
    """
    print("\n" + "="*80)
    print("STEP 2: GROUND TRUTH ANNOTATION ANALYSIS")
    print("="*80)
    print("NOTE: These statistics are for YOUR LABELED SPANS (ground truth)")
    print("      NOT for suggester output")
    print("="*80)

    stats = {
        'span_token_lengths': [],      # Token length of each span
        'doc_token_lengths': [],       # Token length of each document
        'misaligned': [],              # Spans that don't align with tokens
        'whitespace_issues': [],       # Spans with leading/trailing whitespace
        'newline_issues': [],          # Spans crossing line breaks
        'label_counts': Counter(),     # Count of each label type
        'per_label_tokens': defaultdict(list),  # Token lengths per label
        'transactions_with_spans': 0,  # Transactions that have at least one span
        'transactions_by_span_length': defaultdict(int)  # Count of transactions with N-token spans
    }

    # Generator for pipe
    texts = (t for t, _ in parsed_data)
    
    print("\nRunning spaCy tokenizer (pipeline disabled for speed)...")
    docs = nlp.pipe(texts, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

    for (text, entities), doc in zip(parsed_data, docs):
        stats['doc_token_lengths'].append(len(doc))
        
        if entities:
            stats['transactions_with_spans'] += 1
            
            # Track what token lengths appear in this transaction
            transaction_span_lengths = set()
            
            for ent in entities:
                stats['label_counts'][ent['label']] += 1
                
                # 1. Whitespace Check
                if ent['text'].strip() != ent['text']:
                    stats['whitespace_issues'].append({
                        'text': ent['text'],
                        'label': ent['label'],
                        'context': text[:100]
                    })

                # 2. Newline Check
                if '\n' in ent['text']:
                    stats['newline_issues'].append({
                        'text': ent['text'],
                        'label': ent['label'],
                        'context': text[:100]
                    })

                # 3. Alignment & Token Length
                span = doc.char_span(ent['start'], ent['end'], label=ent['label'])
                
                if span is None:
                    # Try rescue modes
                    contract_span = doc.char_span(ent['start'], ent['end'], alignment_mode="contract")
                    expand_span = doc.char_span(ent['start'], ent['end'], alignment_mode="expand")
                    
                    stats['misaligned'].append({
                        'text': ent['text'],
                        'label': ent['label'],
                        'full_context': text,
                        'start': ent['start'],
                        'end': ent['end'],
                        'contract_rescuable': contract_span is not None,
                        'expand_rescuable': expand_span is not None
                    })
                else:
                    token_len = len(span)
                    stats['span_token_lengths'].append(token_len)
                    stats['per_label_tokens'][ent['label']].append(token_len)
                    transaction_span_lengths.add(token_len)
            
            # Record what span lengths appeared in this transaction
            for span_len in transaction_span_lengths:
                stats['transactions_by_span_length'][span_len] += 1

    # ========================================================================
    # REPORTING
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 SPAN TOKEN LENGTH STATISTICS (GROUND TRUTH)")
    print("="*80)
    
    if stats['span_token_lengths']:
        span_lens = stats['span_token_lengths']
        total_spans = len(span_lens)
        
        print(f"\n✅ Successfully Aligned Spans: {total_spans:,}")
        print(f"   Min Tokens:  {min(span_lens)}")
        print(f"   Max Tokens:  {max(span_lens)}")
        print(f"   Mean Tokens: {np.mean(span_lens):.2f}")
        print(f"   Median:      {np.median(span_lens):.1f}")
        print(f"   95th %ile:   {np.percentile(span_lens, 95):.1f}")
        print(f"   99th %ile:   {np.percentile(span_lens, 99):.1f}")
        
        # Span Length Distribution
        print(f"\n📈 SPAN LENGTH DISTRIBUTION (by count of spans):")
        print(f"{'Length':<12} {'Count':<12} {'% of Spans':<15} {'Bar Chart'}")
        print("-" * 80)
        
        span_dist = Counter(span_lens)
        for length in sorted(span_dist.keys())[:15]:  # Show top 15 lengths
            count = span_dist[length]
            pct = (count / total_spans) * 100
            bar = '█' * int(pct / 2)  # Scale to fit
            print(f"{length:<12} {count:<12,} {pct:<14.2f}% {bar}")
        
        if len(span_dist) > 15:
            remaining = sum(span_dist[k] for k in sorted(span_dist.keys())[15:])
            pct = (remaining / total_spans) * 100
            print(f"{'16+':<12} {remaining:<12,} {pct:<14.2f}%")
        
        # Transaction-Level Distribution
        print(f"\n📊 TRANSACTION-LEVEL DISTRIBUTION:")
        print(f"   (What % of transactions contain spans of N tokens?)")
        print(f"{'Length':<12} {'Transactions':<15} {'% of All Txns':<20} {'Bar Chart'}")
        print("-" * 80)
        
        total_txns = len([d for d in parsed_data if d[1]])  # Transactions with spans
        
        for length in sorted(stats['transactions_by_span_length'].keys())[:15]:
            txn_count = stats['transactions_by_span_length'][length]
            pct = (txn_count / total_txns) * 100
            bar = '█' * int(pct / 2)
            print(f"{length:<12} {txn_count:<15,} {pct:<19.2f}% {bar}")
        
        if len(stats['transactions_by_span_length']) > 15:
            remaining = sum(stats['transactions_by_span_length'][k] 
                          for k in sorted(stats['transactions_by_span_length'].keys())[15:])
            pct = (remaining / total_txns) * 100
            print(f"{'16+':<12} {remaining:<15,} {pct:<19.2f}%")
    
    else:
        print("❌ No valid spans found!")
    
    # Per-Label Statistics
    print(f"\n📋 PER-LABEL TOKEN STATISTICS:")
    print(f"{'Label':<20} {'Count':<10} {'Min':<6} {'Max':<6} {'Mean':<8}")
    print("-" * 80)
    for label, token_lens in sorted(stats['per_label_tokens'].items(), 
                                    key=lambda x: -len(x[1]))[:10]:
        print(f"{label:<20} {len(token_lens):<10,} {min(token_lens):<6} "
              f"{max(token_lens):<6} {np.mean(token_lens):<8.2f}")
    
    # Data Quality Issues
    print(f"\n" + "="*80)
    print("🗑️ DATA QUALITY ISSUES")
    print("="*80)
    
    total_spans_attempted = sum(len(entities) for _, entities in parsed_data)
    misaligned_count = len(stats['misaligned'])
    misalignment_rate = (misaligned_count / total_spans_attempted * 100) if total_spans_attempted else 0
    
    print(f"\n⚠️ MISALIGNMENT ANALYSIS:")
    print(f"   Total spans in dataset:    {total_spans_attempted:,}")
    print(f"   Successfully aligned:      {len(stats['span_token_lengths']):,}")
    print(f"   MISALIGNED (UNUSABLE):     {misaligned_count:,} ({misalignment_rate:.2f}%)")
    
    if misalignment_rate > 5:
        print(f"\n   🔥 CRITICAL: {misalignment_rate:.2f}% misalignment rate!")
        print(f"   → These spans CANNOT be learned by the model")
        print(f"   → See detailed examples below")
    
    print(f"\n   Whitespace issues:         {len(stats['whitespace_issues']):,}")
    print(f"   Newline spans:             {len(stats['newline_issues']):,}")

    # Grouped Misalignment Analysis
    if stats['misaligned']:
        print(f"\n" + "="*80)
        print("🔍 GROUPED MISALIGNMENT ANALYSIS")
        print("="*80)
        
        # Group by label
        by_label = defaultdict(list)
        for mis in stats['misaligned']:
            by_label[mis['label']].append(mis)
        
        print(f"\n📊 MISALIGNMENTS BY LABEL (Top 10):")
        print(f"{'Label':<20} {'Count':<10} {'% of Misaligned':<18} {'Top 2 Examples'}")
        print("-" * 80)
        
        for label, items in sorted(by_label.items(), key=lambda x: -len(x[1]))[:10]:
            count = len(items)
            pct = (count / len(stats['misaligned'])) * 100
            examples = [f"'{item['text'][:20]}...'" if len(item['text']) > 20 else f"'{item['text']}'" 
                       for item in items[:2]]
            examples_str = ", ".join(examples)
            print(f"{label:<20} {count:<10} {pct:<17.2f}% {examples_str[:40]}")
        
        # Group by pattern (colon, slash, hyphen, etc.)
        print(f"\n📊 MISALIGNMENT PATTERNS (Top 5):")
        patterns = {
            'Contains colon (:)': [],
            'Contains slash (/)': [],
            'Contains hyphen (-)': [],
            'Contains asterisk (*)': [],
            'Contains period (.)': [],
            'Other': []
        }
        
        for mis in stats['misaligned']:
            text = mis['text']
            if ':' in text:
                patterns['Contains colon (:)'].append(mis)
            elif '/' in text:
                patterns['Contains slash (/)'].append(mis)
            elif '-' in text:
                patterns['Contains hyphen (-)'].append(mis)
            elif '*' in text:
                patterns['Contains asterisk (*)'].append(mis)
            elif '.' in text:
                patterns['Contains period (.)'].append(mis)
            else:
                patterns['Other'].append(mis)
        
        print(f"{'Pattern':<25} {'Count':<10} {'% of Misaligned':<18} {'Top 2 Examples'}")
        print("-" * 80)
        
        for pattern, items in sorted(patterns.items(), key=lambda x: -len(x[1]))[:5]:
            if items:
                count = len(items)
                pct = (count / len(stats['misaligned'])) * 100
                examples = [f"'{item['text'][:20]}...'" if len(item['text']) > 20 else f"'{item['text']}'" 
                           for item in items[:2]]
                examples_str = ", ".join(examples)
                print(f"{pattern:<25} {count:<10} {pct:<17.2f}% {examples_str[:40]}")

    # Detailed Misalignment Examples
    if stats['misaligned']:
        print(f"\n" + "="*80)
        print("🔍 DETAILED MISALIGNMENT EXAMPLES (First 10)")
        print("="*80)
        
        for i, mis in enumerate(stats['misaligned'][:10], 1):
            doc = nlp.make_doc(mis['full_context'])
            
            print(f"\n❌ MISALIGNED SPAN #{i}")
            print(f"{'='*80}")
            print(f"Label:           {mis['label']}")
            print(f"Expected Text:   '{mis['text']}'")
            print(f"Char Range:      [{mis['start']}:{mis['end']}]")
            print(f"Full Context:    {mis['full_context'][:100]}...")
            print(f"\nTokenization:")
            print(f"   Tokens: {[t.text for t in doc]}")
            print(f"   Token boundaries:")
            for j, token in enumerate(doc):
                print(f"      Token {j}: '{token.text}' (chars {token.idx}-{token.idx + len(token.text)})")
            
            print(f"\nRescue Options:")
            print(f"   Contract mode: {'✅ RESCUABLE' if mis['contract_rescuable'] else '❌ NOT RESCUABLE'}")
            print(f"   Expand mode:   {'✅ RESCUABLE' if mis['expand_rescuable'] else '❌ NOT RESCUABLE'}")
            
            if mis['expand_rescuable']:
                expand_span = doc.char_span(mis['start'], mis['end'], alignment_mode="expand")
                print(f"   → Expand would give: '{expand_span.text}'")
    
    return stats

# ============================================================================
# 3. SUGGESTER VALIDATION (RECALL CEILING)
# ============================================================================

def validate_suggester_coverage(parsed_data, nlp, stats):
    """
    Tests ACTUAL suggester to see if it suggests your gold spans
    """
    if not IS_SPANCAT:
        print("\nSkipping Suggester Test (Not using SpanCat)")
        return {}
        
    print("\n" + "="*80)
    print("STEP 3: SUGGESTER VALIDATION (RECALL CEILING)")
    print("="*80)
    print("Testing if ngram_suggester will suggest your GROUND TRUTH spans")
    print("="*80)

    try:
        from spacy.pipeline.spancat import ngram_suggester
    except ImportError:
        print("❌ Requires spaCy v3.1+ for spancat components")
        return {}

    if not stats['span_token_lengths']:
        print("❌ No valid spans to test (all misaligned).")
        return {}

    # Define Candidate Configs
    span_lens = stats['span_token_lengths']
    min_t = min(span_lens)
    max_t = max(span_lens)
    p95_t = int(np.percentile(span_lens, 95))
    p99_t = int(np.percentile(span_lens, 99))
    
    configs = {
        'Conservative (p95)': list(range(min_t, p95_t + 1)),
        'Aggressive (p99)':   list(range(min_t, p99_t + 1)),
        'Balanced (Max)':     list(range(min_t, max_t + 1)),
        'Default (1-3)':      [1, 2, 3]
    }

    # Test on Sample
    test_size = 200
    print(f"\nTesting on random {test_size} documents...")
    import random
    test_batch = random.sample([d for d in parsed_data if d[1]], 
                               min(test_size, len([d for d in parsed_data if d[1]])))

    best_config = None
    best_score = -1
    results = {}

    print(f"\n{'CONFIG NAME':<25} {'SIZES':<15} {'COVERAGE':<12} {'CANDIDATES/DOC':<18} {'VERDICT'}")
    print("-" * 90)

    for name, sizes in configs.items():
        gold_spans = 0
        covered_spans = 0
        total_candidates = 0
        uncovered_examples = []

        for text, entities in test_batch:
            doc = nlp.make_doc(text)
            
            # Run Suggester
            candidates = ngram_suggester([doc], sizes=sizes)
            cand_set = set()
            if candidates.lengths[0] > 0:
                for i in range(candidates.lengths[0]):
                    s, e = candidates.data[i]
                    cand_set.add((int(s), int(e)))
            
            total_candidates += len(cand_set)
            
            # Check Gold Coverage
            for ent in entities:
                span = doc.char_span(ent['start'], ent['end'])
                if span:
                    gold_spans += 1
                    if (span.start, span.end) in cand_set:
                        covered_spans += 1
                    else:
                        uncovered_examples.append({
                            'text': text,
                            'span_text': ent['text'],
                            'label': ent['label'],
                            'tokens': len(span)
                        })
        
        coverage = (covered_spans / gold_spans * 100) if gold_spans else 0
        avg_cand = total_candidates / len(test_batch)
        
        # Scoring logic
        score = coverage
        if avg_cand > 500: 
            score -= 10
            verdict = "⚠️ Too many candidates"
        elif coverage >= 99:
            verdict = "✅ RECOMMENDED"
        elif coverage >= 95:
            verdict = "✓ Good"
        else:
            verdict = "❌ Low coverage"
        
        if score > best_score:
            best_score = score
            best_config = {
                'name': name, 
                'sizes': sizes, 
                'coverage': coverage, 
                'avg': avg_cand,
                'uncovered': uncovered_examples[:5]
            }
        
        results[name] = {
            'sizes': sizes,
            'coverage': coverage,
            'avg_candidates': avg_cand,
            'verdict': verdict
        }

        size_str = f"{min(sizes)}-{max(sizes)}"
        print(f"{name:<25} {size_str:<15} {coverage:>6.2f}%     {avg_cand:>8.1f}          {verdict}")
    
    # Show uncovered examples
    if best_config and best_config['uncovered']:
        print(f"\n⚠️ EXAMPLES OF SPANS NOT SUGGESTED BY '{best_config['name']}':")
        for i, ex in enumerate(best_config['uncovered'], 1):
            print(f"   {i}. '{ex['span_text']}' ({ex['tokens']} tokens) - Label: {ex['label']}")

    # ========================================================================
    # SPANFINDER COMPARISON (if available)
    # ========================================================================
    if SPANFINDER_AVAILABLE:
        print(f"\n" + "="*80)
        print("SPANFINDER COMPARISON (ML-based suggester)")
        print("="*80)
        print("SpanFinder uses ML to predict span boundaries → fewer candidates")
        print("Use when: spans are long (10+ tokens) OR memory is constrained")
        print("="*80)
        
        # Note: SpanFinder requires training, so we can't test it directly
        # But we can estimate its benefit
        avg_span_length = np.mean(stats['span_token_lengths'])
        max_span_length = max(stats['span_token_lengths'])
        
        if max_span_length > 10 or (best_config and best_config['avg'] > 300):
            print(f"\n✅ SPANFINDER RECOMMENDED for your data:")
            print(f"   • Max span length: {max_span_length} tokens")
            if best_config:
                print(f"   • Ngram suggester generates ~{best_config['avg']:.0f} candidates/doc")
            print(f"   • SpanFinder would generate ~50-150 candidates/doc (estimate)")
            print(f"\n   To use SpanFinder:")
            print(f"   1. pip install spacy-experimental")
            print(f"   2. Use this config:")
            print(f"      ```python")
            print(f"      spancat_config = {{")
            print(f"          'suggester': {{")
            print(f"              '@misc': 'spacy-experimental.span_finder_suggester.v1',")
            print(f"              'threshold': 0.5")
            print(f"          }},")
            print(f"          'spans_key': '{SPANCAT_KEY}'")
            print(f"      }}")
            print(f"      ```")
        else:
            print(f"\n✓ NGRAM SUGGESTER IS SUFFICIENT:")
            print(f"   • Max span length: {max_span_length} tokens (reasonable)")
            if best_config:
                print(f"   • Avg candidates: ~{best_config['avg']:.0f}/doc (acceptable)")
            print(f"   • SpanFinder adds complexity without significant benefit")
    
    return best_config, results

# ============================================================================
# 4. FINAL RECOMMENDATIONS & EXPORT
# ============================================================================

def generate_recommendations(parsed_data, nlp, stats, best_config, results):
    """Generate final recommendations and save report"""
    print("\n" + "="*80)
    print("🎯 FINAL RECOMMENDATIONS")
    print("="*80)
    
    report_lines = []
    report_lines.append("SPACY NER TRAINING DATA ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Dataset: {TRAIN_TABLE}")
    report_lines.append(f"Total Transactions: {len(parsed_data):,}")
    report_lines.append(f"Transactions with Spans: {stats['transactions_with_spans']:,}")
    report_lines.append("")
    
    # Data Quality Summary
    total_spans = sum(len(entities) for _, entities in parsed_data)
    aligned_spans = len(stats['span_token_lengths'])
    misaligned_spans = len(stats['misaligned'])
    
    report_lines.append("DATA QUALITY SUMMARY")
    report_lines.append("-"*80)
    report_lines.append(f"Total Spans:              {total_spans:,}")
    report_lines.append(f"Successfully Aligned:     {aligned_spans:,} ({aligned_spans/total_spans*100:.2f}%)")
    report_lines.append(f"Misaligned (UNUSABLE):    {misaligned_spans:,} ({misaligned_spans/total_spans*100:.2f}%)")
    report_lines.append(f"Whitespace Issues:        {len(stats['whitespace_issues']):,}")
    report_lines.append(f"Newline Issues:           {len(stats['newline_issues']):,}")
    report_lines.append("")
    
    # Advanced Metrics
    report_lines.append("ADVANCED QUALITY METRICS")
    report_lines.append("-"*80)
    if stats.get('sd_score'):
        report_lines.append(f"Span Distinctiveness:     {stats['sd_score']:.3f}")
        if stats['sd_score'] > 1.0:
            report_lines.append("  → Entities are distinctive (good for training)")
        else:
            report_lines.append("  → Entities blend with regular text (harder task)")
    
    if stats.get('space_token_count') is not None:
        report_lines.append(f"Docs with Space Tokens:   {stats['space_token_count']:,}")
        if stats['space_token_count'] > 0:
            report_lines.append("  ⚠️ JSON offsets MUST account for space tokens!")
    
    if stats.get('alignment_issues'):
        perfect = stats['alignment_issues']['perfect']
        cuts_through = len(stats['alignment_issues']['cuts_through_token'])
        missing_space = len(stats['alignment_issues']['missing_space_token'])
        total_checked = perfect + cuts_through + missing_space + len(stats['alignment_issues']['boundary_mismatch'])
        if total_checked > 0:
            report_lines.append(f"Offset Verification:      {perfect}/{total_checked} perfect ({perfect/total_checked*100:.1f}%)")
    report_lines.append("")
    
    # Span Statistics
    if stats['span_token_lengths']:
        span_lens = stats['span_token_lengths']
        report_lines.append("SPAN TOKEN LENGTH STATISTICS")
        report_lines.append("-"*80)
        report_lines.append(f"Min:    {min(span_lens)}")
        report_lines.append(f"Max:    {max(span_lens)}")
        report_lines.append(f"Mean:   {np.mean(span_lens):.2f}")
        report_lines.append(f"Median: {np.median(span_lens):.1f}")
        report_lines.append(f"95th:   {np.percentile(span_lens, 95):.1f}")
        report_lines.append(f"99th:   {np.percentile(span_lens, 99):.1f}")
        report_lines.append("")
    
    # Suggester Recommendation
    if IS_SPANCAT and best_config:
        report_lines.append("SUGGESTER CONFIGURATION")
        report_lines.append("-"*80)
        report_lines.append(f"Recommended: {best_config['name']}")
        report_lines.append(f"Sizes: {min(best_config['sizes'])}-{max(best_config['sizes'])}")
        report_lines.append(f"Coverage: {best_config['coverage']:.2f}%")
        report_lines.append(f"Avg Candidates/Doc: {best_config['avg']:.1f}")
        report_lines.append("")
        report_lines.append("Python Config:")
        report_lines.append("```python")
        report_lines.append("spancat_config = {")
        report_lines.append("    'suggester': {")
        report_lines.append("        '@misc': 'spacy.ngram_range_suggester.v1',")
        report_lines.append(f"        'min_size': {min(best_config['sizes'])},")
        report_lines.append(f"        'max_size': {max(best_config['sizes'])}")
        report_lines.append("    },")
        report_lines.append(f"    'spans_key': '{SPANCAT_KEY}',")
        report_lines.append("    'threshold': 0.5")
        report_lines.append("}")
        report_lines.append("```")
        report_lines.append("")
        
        print(f"\n✅ RECOMMENDED SUGGESTER: {best_config['name']}")
        print(f"   Config: min_size={min(best_config['sizes'])}, max_size={max(best_config['sizes'])}")
        print(f"   Coverage: {best_config['coverage']:.2f}%")
        print(f"   Avg Candidates: {best_config['avg']:.1f} per doc")
        
        print(f"\n📋 PASTE THIS INTO YOUR TRAINING CODE:")
        print(f"```python")
        print(f"spancat_config = {{")
        print(f"    'suggester': {{")
        print(f"        '@misc': 'spacy.ngram_range_suggester.v1',")
        print(f"        'min_size': {min(best_config['sizes'])},")
        print(f"        'max_size': {max(best_config['sizes'])}")
        print(f"    }},")
        print(f"    'spans_key': '{SPANCAT_KEY}',")
        print(f"    'threshold': 0.5")
        print(f"}}")
        print(f"```")
    
    # Action Items
    report_lines.append("ACTION ITEMS")
    report_lines.append("-"*80)
    
    misalignment_rate = (misaligned_spans / total_spans * 100) if total_spans else 0
    
    if misalignment_rate > 10:
        action = "1. 🔥 CRITICAL: Fix misalignment issues (>10% of spans unusable)"
        report_lines.append(action)
        print(f"\n{action}")
        print(f"   Options:")
        print(f"   A. Use alignment_mode='expand' in training")
        print(f"   B. Re-annotate the {misaligned_spans:,} misaligned spans")
        print(f"   C. Filter them out (nuclear option)")
    elif misalignment_rate > 5:
        action = "1. ⚠️ Address misalignment issues (5-10% of spans)"
        report_lines.append(action)
        print(f"\n{action}")
    else:
        action = "1. ✅ Misalignment rate acceptable (<5%)"
        report_lines.append(action)
        print(f"\n{action}")
    
    if IS_SPANCAT and best_config:
        action = f"2. ✅ Use recommended suggester config ({best_config['name']})"
        report_lines.append(action)
        print(f"{action}")
    
    report_lines.append("3. Review per-label statistics for class imbalance")
    report_lines.append("4. Consider data augmentation for rare labels")
    
    print(f"\n3. Review per-label statistics for class imbalance")
    print(f"4. Consider data augmentation for rare labels")
    
    # Save Report
    report_text = "\n".join(report_lines)
    with open("/dbfs/tmp/spacy_eda_report.txt", "w") as f:
        f.write(report_text)
    print(f"\n" + "="*80)
    print(f"📄 Full report saved to: /dbfs/tmp/spacy_eda_report.txt")
    print(f"="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete EDA pipeline"""
    print("\n" + "="*80)
    print("SPACY NER TRAINING DATA ANALYZER")
    print("="*80)
    print(f"Model: {BASE_MODEL}")
    print(f"Table: {TRAIN_TABLE}")
    print(f"SpanCat Mode: {IS_SPANCAT}")
    print("="*80)
    
    # Load data
    data, nlp, df = load_and_parse()
    
    if not data:
        print("❌ No data loaded. Exiting.")
        return
    
    # ========================================================================
    # STEP 1.5: ADVANCED QUALITY CHECKS
    # ========================================================================
    
    # Span Distinctiveness calculation
    sd_score = calculate_span_distinctiveness(data, nlp, sample_size=1000)
    
    # Space token detection (CRITICAL for alignment)
    space_token_docs = detect_space_tokens(data, nlp, max_examples=5)
    
    # Offset alignment verification
    alignment_issues = verify_offset_alignment(data, nlp, sample_size=200)
    
    # ========================================================================
    # STEP 2: COMPREHENSIVE TOKENIZATION ANALYSIS
    # ========================================================================
    
    stats = analyze_tokenization_comprehensive(data, nlp)
    
    # Add SD score to stats for later reference
    stats['sd_score'] = sd_score
    stats['space_token_count'] = len(space_token_docs)
    stats['alignment_issues'] = alignment_issues
    
    # ========================================================================
    # STEP 3: SUGGESTER VALIDATION
    # ========================================================================
    
    best_cfg, results = validate_suggester_coverage(data, nlp, stats)
    
    # Final recommendations
    generate_recommendations(data, nlp, stats, best_cfg, results)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
