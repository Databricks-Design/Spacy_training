

import pandas as pd
import numpy as np
import json
import spacy
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set display options for clean tables
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Initialize blank spaCy model for accurate tokenization
# We need this to measure "Token Length" even if we don't save to DocBin
try:
    nlp = spacy.blank("en")
except:
    print("Installing spacy...")
    import sys
    !{sys.executable} -m pip install spacy
    nlp = spacy.blank("en")

print("="*80)
print("🚀 STARTING DUAL-MODE EDA")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/5] Loading Datasets...")

try:
    # Replace these SQL queries with your actual table names
    train_df = spark.sql("").toPandas()
    val_df = spark.sql("SELECed").toPandas()
    print(f" ✓ Training Data: {train_df.shape[0]:,} rows")
    print(f" ✓ Validation Data: {val_df.shape[0]:,} rows")
except NameError:
    print(" ⚠ Spark not detected. Please ensure 'train_df' and 'val_df' are loaded manually.")
    # train_df = pd.read_csv("train.csv") # Uncomment for local

# ============================================================================
# STEP 2: ENTITY DENSITY (How "rich" is the data?)
# ============================================================================
# This answers: "What % of my data has 1 entity? 2 entities? etc."

def analyze_density(df, name="TRAIN"):
    print(f"\n🔢 {name} ENTITY DENSITY")
    print("-" * 60)
    
    counts = []
    for x in df['ENTITIES_LABEL']:
        try:
            ents = json.loads(x)
            counts.append(len(ents))
        except:
            counts.append(0)
            
    density = Counter(counts)
    total = len(counts)
    
    print(f"{'# Entities':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 40)
    
    for k in sorted(density.keys()):
        pct = (density[k] / total) * 100
        print(f"{k:<15} {density[k]:>10,} {pct:>11.2f}%")
        
    print(f"\n   Avg Entities/Row: {np.mean(counts):.2f}")

analyze_density(train_df, "TRAINING")
analyze_density(val_df, "VALIDATION")

# ============================================================================
# STEP 3: OVERLAP CONFLICT ANALYSIS (NER vs SpanCat Decision)
# ============================================================================
# This answers: "Which labels are fighting for the same tokens?"

def analyze_overlaps(df, name="TRAIN"):
    print(f"\n⚔️ {name} OVERLAP CONFLICT MATRIX")
    print("(If a pair appears here, you CANNOT use standard NER for them)")
    print("-" * 80)
    
    conflict_pairs = Counter()
    overlapping_docs = 0
    total_docs = 0
    
    for idx, row in df.iterrows():
        text = row['TEXT']
        if not isinstance(text, str) or not text.strip(): continue
        
        try:
            ents = json.loads(row['ENTITIES_LABEL'])
        except: continue
        
        if not ents: continue
        total_docs += 1
        
        # Simple overlap check
        # We track which label claims which character index
        char_claims = defaultdict(list)
        has_overlap = False
        
        for e in ents:
            for i in range(e['start_char'], e['end_char']):
                char_claims[i].append(e['label'])
                
        # Check claims
        seen_pairs_in_doc = set()
        for char_idx, labels in char_claims.items():
            if len(labels) > 1:
                has_overlap = True
                # Register the conflict pair
                unique_labels = sorted(list(set(labels)))
                if len(unique_labels) > 1:
                    import itertools
                    for pair in itertools.combinations(unique_labels, 2):
                        if pair not in seen_pairs_in_doc:
                            conflict_pairs[pair] += 1
                            seen_pairs_in_doc.add(pair)
                            
        if has_overlap:
            overlapping_docs += 1

    # Report
    if overlapping_docs == 0:
        print("   ✅ NO OVERLAPS FOUND. Safe for Standard NER.")
    else:
        print(f"   🚨 OVERLAPS DETECTED: {overlapping_docs:,} docs ({overlapping_docs/total_docs*100:.2f}%)")
        print("   These labels MUST go to SpanCat (or be merged):")
        print(f"   {'Label A':<25} {'Label B':<25} {'Conflict Count':>15}")
        print("   " + "-"*65)
        for pair, count in conflict_pairs.most_common(10):
            print(f"   {pair[0]:<25} {pair[1]:<25} {count:>15,}")

analyze_overlaps(train_df, "TRAINING")

# ============================================================================
# STEP 4: SPAN LENGTH PROFILING (Suggester Config)
# ============================================================================
# This answers: "How large should my n-gram suggester be?"

def profile_spans(df, name="TRAIN"):
    print(f"\n📏 {name} SPAN LENGTH PROFILING (Tokens)")
    print("-" * 80)
    
    lengths = []
    for idx, row in df.iterrows():
        try:
            text = row['TEXT']
            if not text: continue
            ents = json.loads(row['ENTITIES_LABEL'])
            doc = nlp.make_doc(text) # Tokenize
            for e in ents:
                span = doc.char_span(e['start_char'], e['end_char'], alignment_mode="contract")
                if span:
                    lengths.append(len(span))
        except: pass
        
    if not lengths:
        print("   No valid spans found.")
        return

    # Calculate coverage
    len_counts = Counter(lengths)
    total = sum(len_counts.values())
    sorted_lens = sorted(len_counts.keys())
    
    cum_pct = 0
    suggested_sizes = []
    
    print(f"   {'Length':<10} {'Count':>10} {'Coverage':>12}")
    for l in sorted_lens:
        c = len_counts[l]
        pct = (c/total)*100
        cum_pct += pct
        print(f"   {l:<10} {c:>10,} {cum_pct:>11.1f}%")
        
        # Suggest size if we haven't reached 99.5% coverage yet
        if cum_pct <= 99.5 or (cum_pct - pct) < 99.5:
            suggested_sizes.append(l)
            
    print(f"\n   ✅ SUGGESTED CONFIG: sizes = {suggested_sizes}")

profile_spans(train_df, "TRAINING")

# ============================================================================
# STEP 5: LABEL IMBALANCE & VOLUME
# ============================================================================
# This answers: "Do I have enough data for each label?"

def check_imbalance(df, name="TRAIN"):
    print(f"\n⚖️ {name} LABEL IMBALANCE")
    print("-" * 80)
    
    counts = Counter()
    for x in df['ENTITIES_LABEL']:
        try:
            ents = json.loads(x)
            for e in ents:
                counts[e['label']] += 1
        except: pass
        
    print(f"   {'Label':<30} {'Count':>10} {'Share %':>10} {'Status':<15}")
    print("   " + "-"*70)
    
    total_ents = sum(counts.values())
    
    for label, count in counts.most_common():
        share = (count / total_ents) * 100
        status = "Good"
        if count < 50: status = "CRITICAL (<50)"
        elif count < 200: status = "Low (<200)"
        elif share > 50: status = "Dominant (>50%)"
        
        print(f"   {label:<30} {count:>10,} {share:>9.1f}% {status:<15}")

check_imbalance(train_df, "TRAINING")

print("\n" + "="*80)
print("✅ EDA COMPLETE")
print("="*80)
