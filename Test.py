import pandas as pd
import json
import re
from collections import Counter
import numpy as np
import io
import math
from difflib import SequenceMatcher

# --- 1. Define Regex Patterns for ALL Original Rules ---

# --- High-Precision Extraction Patterns (ext) ---
PATTERN_EXT1 = re.compile(r'"(.+?)"') # Quotes
PATTERN_EXT4 = re.compile(r'\b(INC\.|LLC|CO\.|COM|LTD|CORP|P\.\s*C\.|PC)\b', re.IGNORECASE) # Suffixes

# --- Cleaning Patterns (sim) ---
PATTERN_SIM1_SHORTHAND = re.compile(r'(debit card|credit card|debit|credit|card|crd|ref|cashier check purchase|paypal|NY|New York|Las Vegas|NV|San Francisco|SF|San Francis|San Mateo|San Jose|Port Melbourn|CA|JAMAICA|Sydney|NS|Log Angeles|AU|Surry Hills|Singapore|SG)', re.IGNORECASE)
PATTERN_SIM2_ALPHANUM = re.compile(r'\d{5,}|[a-z]+\d+\w*|\d+[a-z]+\w*', re.IGNORECASE) # 5+ digits, or mixed alphanumeric
PATTERN_SIM3_DATE_ID = re.compile(r'\d\d/\d\d|ref[\d^\s]*|crd[\d^\s]*', re.IGNORECASE) # MM/DD, REF..., CRD...
PATTERN_SIM4_TRANSFER = re.compile(r'(internet transfer|online transfer)', re.IGNORECASE)
PATTERN_SIM5_ASTERISK = re.compile(r'\*')
PATTERN_SIM6_PUNCT = re.compile(r'[",.]') # Comma, quote, period

# --- 2. Data Loading Functions ---

def load_data(is_validation=False):
    """
    Placeholder: Loads the data. Must be replaced with actual loading logic.
    If is_validation=True, it loads the validation set (28,214 records).
    """
    if is_validation:
        print("\nLoading Validation Data...")
        # --- MOCK VALIDATION DATA (Adjust slightly to show difference) ---
        mock_data = """
TEXT|ENTITIES_LABEL|BUSINESS_LABEL|LOCATION_LABEL
PURCHASE "AMAZON" LLC REF123|[{ "start_char": 10, "end_char": 16, "label": "BUSINESS", "standardized_name": "Amazon" }] |Amazon|
DEBIT ATM - BANK OF AMER |[{ "start_char": 10, "end_char": 23, "label": "FINANCIAL_INSTITUTION", "standardized_name": "Bank of America" }] ||
PAYPAL *VendorA* VendorB - CA|[{ "start_char": 7, "end_char": 14, "label": "BUSINESS", "standardized_name": "Vendor A" }, { "start_char": 24, "end_char": 26, "label": "LOCATION", "standardized_name": "CA" }] |Vendor A|CA
ONLINE TRANSFER 01/01|[]||
SHORT MEMO|[{ "start_char": 0, "end_char": 10, "label": "BUSINESS", "standardized_name": "Short Memo" }] |Short Memo|
        """
    else:
        print("\nLoading Training Data...")
        # --- MOCK TRAINING DATA ---
        mock_data = """
TEXT|ENTITIES_LABEL|BUSINESS_LABEL|LOCATION_LABEL
CHECKCARD WM SUPERCENTER BIG SPRING TX|[{ "start_char": 10, "end_char": 24, "label": "BUSINESS", "standardized_name": "Walmart" }, { "start_char": 25, "end_char": 35, "label": "LOCATION", "standardized_name": "BIG SPRING" }] |Walmart|BIG SPRING||TX
PAYPAL *VendorA* VendorB - CA|[{ "start_char": 7, "end_char": 14, "label": "BUSINESS", "standardized_name": "Vendor A" }, { "start_char": 24, "end_char": 26, "label": "LOCATION", "standardized_name": "CA" }] |Vendor A|CA
PURCHASE "AMAZON" LLC REF123|[{ "start_char": 10, "end_char": 16, "label": "BUSINESS", "standardized_name": "Amazon" }] |Amazon|
DEBIT ATM - BANK OF AMER |[{ "start_char": 10, "end_char": 23, "label": "FINANCIAL_INSTITUTION", "standardized_name": "Bank of America" }] ||
SHORT MEMO|[{ "start_char": 0, "end_char": 10, "label": "BUSINESS", "standardized_name": "Short Memo" }] |Short Memo|
REPEATED REPEATED REF123|[{ "start_char": 0, "end_char": 8, "label": "BUSINESS", "standardized_name": "Repeated" }] |Repeated|
INTERNET TRANSFER 01/01|[]||
LONG VENDOR INC - NEW YORK|[{ "start_char": 0, "end_char": 13, "label": "BUSINESS", "standardized_name": "Long Vendor Inc" }, { "start_char": 16, "end_char": 24, "label": "LOCATION", "standardized_name": "NEW YORK" }] |Long Vendor Inc|NEW YORK
        """

    df = pd.read_csv(io.StringIO(mock_data), sep='|')
    
    # Required preparation steps
    df['TEXT'] = df['TEXT'].fillna('').astype(str)
    df['spans'] = df['ENTITIES_LABEL'].apply(lambda x: json.loads(x) if pd.notna(x) and x and isinstance(x, str) else [])
    df['num_entities'] = df['spans'].apply(len)
    
    return df

# --- 3. Analysis Functions (Abstracted) ---

def check_entity_match(row, pattern, entity_label=None):
    """Checks if a record contains a pattern match AND optionally a specific entity."""
    text = row['TEXT']
    if not pattern.search(text):
        return False
    if entity_label is None:
        return True
    return any(e['label'] == entity_label for e in row['spans'])

def check_repetition(text, threshold=0.8):
    """Approximates ext2 (Repeated Words) by checking self-similarity."""
    text_list = text.split()
    if len(text_list) < 2: return False
    mid = len(text_list) // 2
    part1 = ' '.join(text_list[:mid])
    part2 = ' '.join(text_list[mid:])
    if len(part1) < 4 or len(part2) < 4: return False
    ratio = SequenceMatcher(None, part1, part2).ratio()
    return ratio > threshold

def check_unique_entity_coverage(df, pattern, entity_label):
    """Calculates unique entity coverage for ext1 and ext4."""
    covered_names = set()
    for index, row in df.iterrows():
        if check_entity_match(row, pattern, entity_label):
            for e in row['spans']:
                if e['label'] == entity_label:
                    covered_names.add(e['standardized_name'])
    return covered_names

def run_analysis_suite(df, name):
    """Runs the core analysis and returns key metrics for comparison."""
    total_records = len(df)
    
    # --- A. PATTERN NECESSITY ---
    patterns_to_check = {
        'sim1 (Shorthand)': PATTERN_SIM1_SHORTHAND,
        'sim4 (Transfer)': PATTERN_SIM4_TRANSFER,
        'sim6 (Punct)': PATTERN_SIM6_PUNCT
    }
    pattern_freqs = {n: df['TEXT'].apply(lambda x: bool(p.search(x))).sum() / total_records for n, p in patterns_to_check.items()}
    
    # --- B. BUSINESS COVERAGE BASELINE (ext1, ext4) ---
    unique_business_names = set()
    df['BUSINESS_LABEL'].dropna().apply(lambda x: unique_business_names.update(x.split('||')))
    total_unique_business = len(unique_business_names)
    
    ext1_names = check_unique_entity_coverage(df, PATTERN_EXT1, 'BUSINESS')
    ext4_names = check_unique_entity_coverage(df, PATTERN_EXT4, 'BUSINESS')
    baseline_coverage_count = len(ext1_names.union(ext4_names))
    
    # --- C. COMPLEXITY AND DELIMITER FREQUENCY ---
    two_entity_df = df[df['num_entities'] == 2].copy()
    
    delimiter_counts = Counter()
    for index, row in two_entity_df.iterrows():
        spans = sorted(row['spans'], key=lambda x: x['start_char'])
        if len(spans) == 2:
            delimiter_text = row['TEXT'][spans[0]['end_char']:spans[1]['start_char']]
            cleaned_delimiter = delimiter_text.strip()
            if cleaned_delimiter == '*': delimiter_counts['*'] += 1
            if cleaned_delimiter == '-': delimiter_counts['-'] += 1
            if cleaned_delimiter == ' - ': delimiter_counts[' - '] += 1

    # Get LOCATION Noise Check
    total_asterisk_records = df['TEXT'].apply(lambda x: bool(PATTERN_SIM5_ASTERISK.search(str(x)))).sum()
    
    return {
        'name': name,
        'records': total_records,
        'multi_entity_rate': df[df['num_entities'] >= 2].shape[0] / total_records,
        'business_coverage_rate': baseline_coverage_count / total_unique_business if total_unique_business else 0,
        'asterisk_delimiter_rate': delimiter_counts['*'] / len(two_entity_df) if len(two_entity_df) else 0,
        'dash_delimiter_rate': (delimiter_counts['-'] + delimiter_counts[' - ']) / len(two_entity_df) if len(two_entity_df) else 0,
        'sim1_freq': pattern_freqs['sim1 (Shorthand)'],
        'sim4_freq': pattern_freqs['sim4 (Transfer)'],
        'asterisk_total_freq': total_asterisk_records / total_records
    }

# --- 4. Main Execution Block ---

def main():
    
    # 1. Run Analysis on TRAINING Data
    df_train = load_data(is_validation=False)
    results_train = run_analysis_suite(df_train, "Training")
    
    # 2. Run Analysis on VALIDATION Data
    df_val = load_data(is_validation=True)
    results_val = run_analysis_suite(df_val, "Validation")

    # 3. Print Comparison
    print("\n" + "#" * 70)
    print("## 🔍 Validation Logic: Training vs. Validation Stability Check")
    print("#" * 70)
    
    print("\n### 1. High-Level Dataset Comparison:")
    print(f"| Dataset | Records | Multi-Entity Rate (>=2) |")
    print("|:--- | ---:| ---:|")
    print(f"| Training | {results_train['records']:,} | {results_train['multi_entity_rate'] * 100:.2f}% |")
    print(f"| Validation | {results_val['records']:,} | {results_val['multi_entity_rate'] * 100:.2f}% |")
    print("\n**Conclusion:** The Multi-Entity Rate (complexity) should be close (e.g., within 5 percentage points) across both sets.")

    print("\n### 2. Rule Coverage and Frequency Stability:")
    print("This table shows if the rules found on the training set hold true on the validation set.")
    print("| Metric | Training Value | Validation Value | Delta (Validation - Training) | Stability |")
    print("|:--- | ---:| ---:| ---:|:--- |")
    
    # BUSINESS COVERAGE
    print(f"| **BUSINESS Baseline Coverage (ext1/ext4)** | {results_train['business_coverage_rate'] * 100:.2f}% | {results_val['business_coverage_rate'] * 100:.2f}% | {results_val['business_coverage_rate'] * 100 - results_train['business_coverage_rate'] * 100:.2f}% | {'STABLE' if abs(results_val['business_coverage_rate'] - results_train['business_coverage_rate']) < 0.05 else 'UNSTABLE'} |")
    
    # DELIMITER FREQUENCY
    print(f"| Asterisk (*) Delimiter Rate | {results_train['asterisk_delimiter_rate'] * 100:.2f}% | {results_val['asterisk_delimiter_rate'] * 100:.2f}% | {results_val['asterisk_delimiter_rate'] * 100 - results_train['asterisk_delimiter_rate'] * 100:.2f}% | {'STABLE' if abs(results_val['asterisk_delimiter_rate'] - results_train['asterisk_delimiter_rate']) < 0.05 else 'UNSTABLE'} |")
    print(f"| Dash (-) Delimiter Rate | {results_train['dash_delimiter_rate'] * 100:.2f}% | {results_val['dash_delimiter_rate'] * 100:.2f}% | {results_val['dash_delimiter_rate'] * 100 - results_train['dash_delimiter_rate'] * 100:.2f}% | {'STABLE' if abs(results_val['dash_delimiter_rate'] - results_train['dash_delimiter_rate']) < 0.05 else 'UNSTABLE'} |")

    # CLEANING NECESSITY
    print(f"| Asterisk (*) Total Frequency | {results_train['asterisk_total_freq'] * 100:.2f}% | {results_val['asterisk_total_freq'] * 100:.2f}% | {results_val['asterisk_total_freq'] * 100 - results_train['asterisk_total_freq'] * 100:.2f}% | {'STABLE' if abs(results_val['asterisk_total_freq'] - results_train['asterisk_total_freq']) < 0.05 else 'UNSTABLE'} |")
    print(f"| Shorthand (sim1) Frequency | {results_train['sim1_freq'] * 100:.2f}% | {results_val['sim1_freq'] * 100:.2f}% | {results_val['sim1_freq'] * 100 - results_train['sim1_freq'] * 100:.2f}% | {'STABLE' if abs(results_val['sim1_freq'] - results_train['sim1_freq']) < 0.05 else 'UNSTABLE'} |")
    print(f"| Transfer (sim4) Frequency | {results_train['sim4_freq'] * 100:.2f}% | {results_val['sim4_freq'] * 100:.2f}% | {results_val['sim4_freq'] * 100 - results_train['sim4_freq'] * 100:.2f}% | {'STABLE' if abs(results_val['sim4_freq'] - results_train['sim4_freq']) < 0.05 else 'UNSTABLE'} |")

    print("\n### 3. Final Stability Conclusion:")
    print("If all metrics are marked 'STABLE' (within 5% difference), the rules discovered are robust and can be reliably applied to the live data.")
    print("If any key metric (especially BUSINESS Coverage) is 'UNSTABLE', it indicates that the validation set contains patterns missed by the training set, requiring further EDA on the unstable patterns.")
    
    # Optional: You can visualize this stability using a simple chart .


# --- Execution Block ---

if __name__ == "__main__":
    main()
