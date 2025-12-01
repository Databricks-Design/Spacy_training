import pandas as pd
import json
import re
from collections import Counter
from itertools import combinations
import numpy as np
import io

# --- 1. Define Regex Patterns for Existing Rules ---

# Pattern for ext1: Text within double quotes
PATTERN_EXT1 = re.compile(r'"(.+?)"')
# Pattern for ext4: Corporate Suffixes (preceded by a space or start of string, case-insensitive)
PATTERN_EXT4 = re.compile(r'\b(INC\.|LLC|CO\.|COM|LTD|CORP|P\.\s*C\.|PC)\b', re.IGNORECASE)
# Pattern for sim5: Asterisk noise
PATTERN_SIM5 = re.compile(r'\*')

# --- 2. Data Loading Placeholder ---

def load_data():
    """
    Placeholder function to load the 'mastercard_openbanking_nam.enrich.trainingforengdata_redacted_relabelled' data.
    
    NOTE: Replace this with your actual data loading method (e.g., read_csv, read_json, or database query).
    
    The function must return a Pandas DataFrame with the columns:
    'TEXT', 'ENTITIES_LABEL', 'BUSINESS_LABEL', 'LOCATION_LABEL'.
    """
    
    # --- Example Mock Data for Demonstration (DELETE THIS SECTION) ---
    mock_data = """
TEXT|ENTITIES_LABEL|BUSINESS_LABEL|LOCATION_LABEL
CHECKCARD 0806 WM SUPERCENTER BIG SPRING TX|[{ "start_char": 11, "end_char": 25, "label": "BUSINESS", "standardized_name": "Walmart" }, { "start_char": 26, "end_char": 36, "label": "LOCATION", "standardized_name": "BIG SPRING" }, { "start_char": 37, "end_char": 39, "label": "LOCATION", "standardized_name": "TX" }] |Walmart|BIG SPRING||TX
PAYPAL *Vendor A* Vendor B LLC|[{ "start_char": 7, "end_char": 15, "label": "BUSINESS", "standardized_name": "Vendor A" }, { "start_char": 17, "end_char": 30, "label": "BUSINESS", "standardized_name": "Vendor B" }] |Vendor A||Vendor B||LLC|
PURCHASE AT "AMAZON" Marketplace|[{ "start_char": 15, "end_char": 21, "label": "MARKETPLACE", "standardized_name": "Amazon" }] |Amazon|
DEBIT ATM - BANK OF AMER |[{ "start_char": 10, "end_char": 23, "label": "FINANCIAL_INSTITUTION", "standardized_name": "Bank of America" }] |
Pizza HUT - NEW YORK, NY|[{ "start_char": 0, "end_char": 9, "label": "BUSINESS", "standardized_name": "Pizza Hut" }, { "start_char": 12, "end_char": 20, "label": "LOCATION", "standardized_name": "New York" }, { "start_char": 22, "end_char": 24, "label": "LOCATION", "standardized_name": "NY" }] |Pizza Hut|NEW YORK||NY
    """
    df = pd.read_csv(io.StringIO(mock_data), sep='|')
    # --- END MOCK DATA ---
    
    # Required data preparation step: convert JSON string to actual list
    df['spans'] = df['ENTITIES_LABEL'].apply(lambda x: json.loads(x) if pd.notna(x) and x else [])
    
    # Add helper columns for entity counting
    df['num_entities'] = df['spans'].apply(len)
    df['entity_labels'] = df['spans'].apply(lambda spans: sorted(list(set(e['label'] for e in spans))))
    
    return df

# --- 3. Core Analysis Logic ---

def check_coverage(row, pattern, entity_label):
    """Checks if a record contains a pattern match AND an entity of a specific label."""
    text = row['TEXT']
    spans = row['spans']
    
    # 1. Check if the pattern matches anywhere in the TEXT
    if not pattern.search(text):
        return False, []
    
    # 2. Check if a relevant entity (BUSINESS/LOCATION) is present
    relevant_spans = [e for e in spans if e['label'] == entity_label]
    if not relevant_spans:
        return False, []

    # Simple Check: Did the pattern and entity occur in the same record?
    return True, [e['standardized_name'] for e in relevant_spans]


def main():
    """Executes the full Exploratory Data Analysis (EDA) flow."""
    df = load_data()
    total_records = len(df)
    
    if total_records == 0:
        print("Error: DataFrame is empty. Please check your data loading function.")
        return

    # --- PHASE 1: Entity Frequency and Span Analysis ---
    
    print("## 📊 Phase 1: Entity Frequency and Span Analysis\n")

    # A. Entity Count per Transaction
    entity_counts = df['num_entities'].value_counts().sort_index()
    max_entities = df['num_entities'].max()
    print("### 1. Entity Count Distribution:")
    print(f"| Entities | Count | Percent |")
    print("|:--- | ---:| ---:|")
    
    for count, num in entity_counts.items():
        if count < 5 or count == max_entities:
            print(f"| {count} | {num:,} | {num / total_records * 100:.2f}% |")
        elif count == 5:
            print("| ... | ... | ... |")
            
    print(f"\n**Max Entities Found in a single transaction:** **{max_entities}**\n")

    # B. BUSINESS & LOCATION Co-occurrence
    total_business = df['BUSINESS_LABEL'].astype(bool).sum()
    total_location = df['LOCATION_LABEL'].astype(bool).sum()

    co_occurrence_df = df[df['entity_labels'].apply(lambda x: 'BUSINESS' in x and 'LOCATION' in x)]
    co_occurrence_count = len(co_occurrence_df)

    print("### 2. BUSINESS & LOCATION Co-occurrence:")
    print(f"* Records with **both** BUSINESS and LOCATION: **{co_occurrence_count:,} ({co_occurrence_count / total_records * 100:.2f}%)**")

    # C. Entity Span Analysis (Max Length)
    all_entity_texts = []
    for spans in df['spans']:
        for entity in spans:
            # We assume TEXT is in the dataframe, but for span text, we need the original row
            # For simplicity, we use the standardized_name length for max span, which is close enough.
            text_len = len(entity.get('standardized_name', ''))
            if text_len > 0:
                all_entity_texts.append(entity.get('standardized_name'))
    
    if all_entity_texts:
        max_char_len = max(len(t) for t in all_entity_texts)
        max_word_count = max(len(t.split()) for t in all_entity_texts)
        print(f"* Maximum Word Count of an Entity Span: **{max_word_count}**")
        print(f"* Maximum Character Length of an Entity Span: **{max_char_len}**")

    print("\n" + "---" + "\n")
    
    # --- PHASE 2: Quantifying Coverage of Existing Rules ---

    print("## 📈 Phase 2: Quantifying Coverage of Existing Rules\n")
    
    # A. BUSINESS Entity Coverage
    # Calculate total unique business names only once
    unique_business_names = set()
    df['BUSINESS_LABEL'].dropna().apply(lambda x: unique_business_names.update(x.split('||')))
    total_unique_business = len(unique_business_names)
    
    covered_business_names = set()

    # ext1: Quotes
    ext1_matches = df.apply(lambda row: check_coverage(row, PATTERN_EXT1, 'BUSINESS'), axis=1)
    for matched, names in ext1_matches:
        if matched:
            for name in names: covered_business_names.add(name)
    ext1_coverage = len(covered_business_names)
    
    # ext4: Suffixes
    ext4_matches = df.apply(lambda row: check_coverage(row, PATTERN_EXT4, 'BUSINESS'), axis=1)
    
    ext4_initial_coverage = len(covered_business_names)
    for matched, names in ext4_matches:
        if matched:
            for name in names: covered_business_names.add(name)
    ext4_additional_coverage = len(covered_business_names) - ext4_initial_coverage

    print("### 1. BUSINESS (Vendor) Coverage:")
    print(f"* Total Unique BUSINESS Entities in Dataset: **{total_unique_business:,}**")
    print(f"| Rule | Records Matched | Unique Entities Covered | Coverage Rate |")
    print("|:--- | ---:| ---:| ---:|")
    print(f"| **ext1 (Quotes)** | {ext1_matches.apply(lambda x: x[0]).sum():,} | {ext1_coverage:,} | {ext1_coverage / total_unique_business * 100:.2f}% |")
    print(f"| **ext4 (Suffixes)** | {ext4_matches.apply(lambda x: x[0]).sum():,} | {ext4_additional_coverage:,} | {ext4_additional_coverage / total_unique_business * 100:.2f}% |")
    print(f"| **Total Heuristic Coverage** | N/A | {len(covered_business_names):,} | **{len(covered_business_names) / total_unique_business * 100:.2f}%** |")


    # B. LOCATION Entity Coverage & Noise
    unique_location_names = set()
    df['LOCATION_LABEL'].dropna().apply(lambda x: unique_location_names.update(x.split('||')))
    total_unique_location = len(unique_location_names)

    sim5_covered = 0
    total_asterisk_records = df['TEXT'].apply(lambda x: bool(PATTERN_SIM5.search(str(x)))).sum()

    for index, row in df.iterrows():
        text = str(row['TEXT'])
        
        if PATTERN_SIM5.search(text):
            last_asterisk_index = text.rfind('*')
            
            # Check if any LOCATION entity starts *after* the last asterisk
            for entity in row['spans']:
                if entity['label'] == 'LOCATION' and entity['start_char'] > last_asterisk_index:
                    sim5_covered += 1
                    break

    print("\n### 2. LOCATION Coverage & Noise:")
    print(f"* Total Unique LOCATION Entities in Dataset: {total_unique_location:,}")
    print(f"* Total Records with an Asterisk (*): {total_asterisk_records:,} ({total_asterisk_records / total_records * 100:.2f}%)")
    print(f"| Rule | Count | Noise Elimination Potential |")
    print("|:--- | ---:| ---:|")
    print(f"| **sim5 (Post-Asterisk)** | {sim5_covered:,} | **{sim5_covered / total_unique_location * 100:.2f}%** |")
    
    print("\n" + "---" + "\n")
    
    # --- PHASE 3: Dynamic Delimiter Discovery ---

    print("## ✨ Phase 3: Dynamic Delimiter Pattern Discovery\n")

    # A. Isolate Multi-Entity Records (focus on 2 entities for clean delimiters)
    two_entity_df = df[df['num_entities'] == 2].copy()
    
    all_delimiters = []
    delimiter_examples = []
    
    # Iterate over transactions with exactly two entities
    for index, row in two_entity_df.iterrows():
        spans = sorted(row['spans'], key=lambda x: x['start_char'])
        
        entity_a = spans[0]
        entity_b = spans[1]
        
        # Extract text between the two entities
        delimiter_text = row['TEXT'][entity_a['end_char']:entity_b['start_char']]
        cleaned_delimiter = delimiter_text.strip()
        
        if cleaned_delimiter:
            all_delimiters.append(cleaned_delimiter)
            
            # Store example for human review
            if len(delimiter_examples) < 10 and cleaned_delimiter not in [e.get('delimiter') for e in delimiter_examples]:
                delimiter_examples.append({
                    'delimiter': cleaned_delimiter,
                    'example_text': row['TEXT'],
                    'entities': [entity_a['standardized_name'], entity_b['standardized_name']]
                })

    # B. Delimiter Frequency Analysis
    delimiter_counts = Counter(all_delimiters)

    # Filter for meaningful delimiters (not just a single space or punctuation)
    # Allows single * or - or /
    meaningful_delimiters = {k: v for k, v in delimiter_counts.items() if len(k.strip()) > 1 or k in ['*', '-', '/']}
    top_delimiters = Counter(meaningful_delimiters).most_common(5)

    print("### 1. Top 5 Meaningful Delimiter Patterns (Found in 2-Entity Transactions):")
    print(f"Found {len(all_delimiters):,} total delimiters in {len(two_entity_df):,} two-entity transactions.")
    print(f"| Rank | Delimiter Pattern | Frequency | Percentage of all Delimiters |")
    print("|:---:|:---:|---:|---:|")
    total_delimiters = len(all_delimiters)
    for rank, (delimiter, count) in enumerate(top_delimiters, 1):
        print(f"| {rank} | **'{delimiter}'** | {count:,} | {count / total_delimiters * 100:.2f}% |")

    print("\n### 2. Example Transactions for Human Review:")
    for ex in delimiter_examples:
        print(f"\n- **Delimiter:** **'{ex['delimiter']}'**")
        print(f"  **Text:** {ex['example_text']}")
        print(f"  **Entities:** {ex['entities']}")

    print("\n---")
    print("### ✅ Dynamic Rule Recommendation:")
    print(f"The analysis of 2-entity transactions highlights the need to split memos based on the detected separators.")
    print(f"**DELIMITER-SPLIT RULE:** Define a rule to split the memo on high-frequency, non-entity separators like the top-ranked delimiter: **'{top_delimiters[0][0]}'**. The resulting segments should then be processed independently by the single-vendor extraction heuristics (ext1, ext4, etc.) to boost coverage of multi-entity records.")


# --- 4. Main Execution Block ---

if __name__ == "__main__":
    main()
