"""
Prepare Transaction Data for spaCy NER Training
================================================

This script:
1. Loads transaction data from CSV/Delta table
2. Extracts entities from all label columns (except ENTITIES_LABEL and EXTRACTED_TEXTS_LABEL)
3. Creates character-level offsets for spaCy training
4. Validates token alignment
5. Performs comprehensive EDA
6. Saves to CSV in spaCy format

Author: Sagar
Date: 2025-12-03
"""

import pandas as pd
import spacy
from spacy.training import offsets_to_biluo_tags
from spacy.tokens import Doc
import json
import re
from collections import Counter, defaultdict
import numpy as np
from tqdm import tqdm
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input file path
INPUT_FILE = "trainingforengdata_redacted_relabelled.csv"  # Change to your file path

# Output file path
OUTPUT_FILE = "spacy_training_data.csv"

# Label columns to process (EXCLUDE these two)
EXCLUDE_COLUMNS = ['ENTITIES_LABEL', 'EXTRACTED_TEXTS_LABEL', 'SENTENCE_ID']

# Entity type mapping from column names to labels
LABEL_MAPPING = {
    'BUSINESS_LABEL': 'BUSINESS',
    'LOCATION_LABEL': 'LOCATION',
    'DELIVERY_SERVICE_LABEL': 'DELIVERY_SERVICE',
    'BNPL_PROVIDER_LABEL': 'BNPL_PROVIDER',
    'DIGITAL_WALLET_LABEL': 'DIGITAL_WALLET',
    'FINANCIAL_INSTITUTION_LABEL': 'FINANCIAL_INSTITUTION',
    'PAYMENT_NETWORK_LABEL': 'PAYMENT_NETWORK',
    'PAYMENT_PROCESSOR_LABEL': 'PAYMENT_PROCESSOR',
    'MARKETPLACE_LABEL': 'MARKETPLACE',
    'PLATFORM_LABEL': 'PLATFORM',
    'PERSON_LABEL': 'PERSON',
    'PAYMENT_TYPE_LABEL': 'PAYMENT_TYPE',
    'STANDARDIZED_NAMES_LABEL': None,  # Skip this
    'ENTITY_TYPES_LABEL': None  # Skip this
}

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def preprocess_text(text):
    """
    Clean transaction text before entity extraction.
    
    Steps:
    1. Replace *, /, # with space when between tokens
    2. Remove <> patterns, | with spaces, XXXX+ tokens
    3. Normalize multiple spaces
    """
    import re
    
    if pd.isna(text) or text == '':
        return text
    
    text = str(text)
    
    # Step 1: Replace separators between tokens with space
    # Pattern: alphanumeric + separator + alphanumeric
    text = re.sub(r'([a-zA-Z0-9])\*([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])/([a-zA-Z0-9])', r'\1 \2', text)
    text = re.sub(r'([a-zA-Z0-9])#([a-zA-Z0-9])', r'\1 \2', text)
    
    # Step 2: Removals
    # Remove <> patterns
    text = re.sub(r'<>', '', text)
    
    # Remove | surrounded by spaces
    text = re.sub(r' \| ', ' ', text)
    
    # Remove tokens with 4+ X's (entire token)
    # Match word boundaries to remove entire tokens
    text = re.sub(r'\b\w*X{4,}\w*\b', '', text)
    
    # Step 3: Cleanup
    # Multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


# ============================================================================
# MULTIPROCESSING GLOBALS
# ============================================================================

# Global variable to hold spaCy model (one per process)
_nlp = None

def init_worker():
    """Initialize spaCy once per worker process."""
    global _nlp
    _nlp = spacy.blank("en")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_entity_offsets(text, entity_text):
    """
    Find all character-level offsets for an entity in text, respecting word boundaries.
    Returns list of (start, end) tuples.
    
    Uses regex word boundaries (\b) to ensure:
    - "CA" doesn't match inside "CARD"
    - Only complete token matches are found
    - Handles special characters (e.g., "amzn.com") correctly
    
    Examples:
        "CA" in "CARD CA" -> returns [(5, 7)] NOT [(0, 2)]
        "amzn.com" in "amzn pos amzn.com" -> returns [(9, 17)] NOT [(0, 4)]
        "PAYPAL" in "PAYPAL*WALMART" -> returns [(0, 6)] (boundary after *)
    """
    if pd.isna(entity_text) or pd.isna(text) or entity_text == '':
        return []
    
    offsets = []
    entity_text = str(entity_text).strip()
    text = str(text)
    
    # Escape special regex characters (e.g., "amzn.com" -> "amzn\.com")
    escaped_entity = re.escape(entity_text)
    
    # \b = word boundary (between \w [a-zA-Z0-9_] and non-\w)
    pattern = r'\b' + escaped_entity + r'\b'
    
    # Find all matches with word boundaries (no fallback to partial matches)
    for match in re.finditer(pattern, text):
        offsets.append((match.start(), match.end()))
    
    return offsets


def extract_entities_from_row(row, text_column='TEXT'):
    """
    Extract all entities from label columns in a row (dict).
    Returns tuple: (entities_list, original_offsets_dict)
    - entities_list: list of (start_char, end_char, label, original_value) tuples
    - original_offsets_dict: dict mapping (entity_value, label) -> (original_start, original_end)
    """
    text = row.get(text_column, '')
    if pd.isna(text) or text == '':
        return [], {}
    
    text = str(text)
    entities = []
    
    # Parse ENTITIES_LABEL to get original offsets
    original_offsets = {}  # key: (entity_value, label), value: (start_char, end_char)
    if 'ENTITIES_LABEL' in row and row['ENTITIES_LABEL']:
        try:
            entities_list = json.loads(row['ENTITIES_LABEL'])
            for ent_dict in entities_list:
                standardized_name = ent_dict.get('standardized_name', '').strip()
                label = ent_dict.get('label', '').strip()
                start_char = ent_dict.get('start_char')
                end_char = ent_dict.get('end_char')
                if standardized_name and label and start_char is not None and end_char is not None:
                    original_offsets[(standardized_name, label)] = (start_char, end_char)
        except:
            pass  # If JSON parsing fails, continue without original offsets
    
    for col_name, label in LABEL_MAPPING.items():
        if label is None:  # Skip columns we don't want
            continue
        
        col_value = row.get(col_name, None)
        if col_value is None or pd.isna(col_value) or col_value == '':
            continue
        
        # Handle pipe-separated values
        entity_values = str(col_value).split('||')
        
        for entity_text in entity_values:
            entity_text = entity_text.strip()
            if entity_text == '':
                continue
            
            # Find all offsets for this entity
            offsets = find_entity_offsets(text, entity_text)
            
            for start, end in offsets:
                entities.append((start, end, label, entity_text))  # Include original value
    
    return entities, original_offsets


def categorize_misalignment(text, start, end, doc):
    """
    Categorize the type of misalignment error in plain language.
    """
    # Check bounds
    if start < 0 or end > len(text) or start >= end:
        return "Invalid character positions"
    
    if start >= len(text) or end > len(text):
        return "Position goes beyond text length"
    
    # Find which tokens the span overlaps
    overlapping_tokens = []
    for token in doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        # Check if entity span overlaps with this token
        if not (end <= token_start or start >= token_end):
            overlapping_tokens.append({
                'text': token.text,
                'start': token_start,
                'end': token_end,
                'idx': token.i
            })
    
    if len(overlapping_tokens) == 0:
        return "No matching tokens found"
    
    # Check if start cuts through a token
    first_token = overlapping_tokens[0]
    if start > first_token['start'] and start < first_token['end']:
        return "Cuts through start of entity"
    
    # Check if end cuts through a token
    last_token = overlapping_tokens[-1]
    if end > last_token['start'] and end < last_token['end']:
        return "Cuts through end of entity"
    
    # Check if both cut through
    if (start > first_token['start'] and start < first_token['end']) and \
       (end > last_token['start'] and end < last_token['end']):
        return "Cuts through both start and end"
    
    return "Other boundary issue"


def process_row(row_dict):
    """
    Process a single row for multiprocessing.
    Returns tuple: (row_idx, result_dict, stats_dict)
    """
    global _nlp
    
    idx = row_dict['_idx']
    text_column = row_dict['_text_col']
    
    row_stats = {
        'total_entities': 0,
        'aligned_entities': 0,
        'misaligned_entities': 0,
        'entity_counts': Counter(),
        'span_lengths': defaultdict(list),
        'misalignment_by_type': Counter(),
        'misalignment_by_label': Counter(),
        'has_misalignment': False,
        'only_misalignment': False,
        'misalignment_details': []
    }
    
    text = row_dict.get(text_column, '')
    
    if pd.isna(text) or text == '':
        return (idx, None, row_stats)
    
    text = str(text)
    
    # Extract entities - pass dict instead of Series
    entities_with_values, original_offsets = extract_entities_from_row(row_dict, text_column=text_column)
    
    if len(entities_with_values) == 0:
        return (idx, None, row_stats)
    
    row_stats['total_entities'] = len(entities_with_values)
    
    # Remove duplicates
    entities_with_values = list(set(entities_with_values))
    entities_with_values = sorted(entities_with_values, key=lambda x: x[0])
    
    # Convert to format for validation
    entities_for_validation = [(start, end, label) for start, end, label, orig in entities_with_values]
    original_values = {(start, end, label): orig for start, end, label, orig in entities_with_values}
    
    # Validate alignment using global nlp
    aligned, misaligned, warnings = validate_alignment(text, entities_for_validation, _nlp, original_values, original_offsets)
    
    row_stats['aligned_entities'] = len(aligned)
    row_stats['misaligned_entities'] = len(misaligned)
    
    if len(misaligned) > 0:
        row_stats['has_misalignment'] = True
        if len(aligned) == 0:
            row_stats['only_misalignment'] = True
    
    # Store misalignment details
    for mis in misaligned:
        key = (mis['start'], mis['end'], mis['label'])
        mis['expected_value'] = original_values.get(key, 'UNKNOWN')
        mis['row_idx'] = idx
        mis['full_text'] = text
        
        row_stats['misalignment_by_type'][mis['error_type']] += 1
        row_stats['misalignment_by_label'][mis['label']] += 1
        row_stats['misalignment_details'].append(mis)
    
    # Count entities by label
    for start, end, label in aligned:
        row_stats['entity_counts'][label] += 1
        
        span_len = get_span_length_in_tokens(text, start, end, _nlp)
        if span_len is not None:
            row_stats['span_lengths'][label].append(span_len)
    
    # Only return result if there are aligned entities
    result = None
    if len(aligned) > 0:
        result = {
            'text': text,
            'entities': json.dumps(aligned)
        }
    
    return (idx, result, row_stats)


def get_token_details(text, start, end, doc):
    """
    Get detailed token information for a span.
    """
    tokens_info = []
    
    for token in doc:
        token_start = token.idx
        token_end = token.idx + len(token.text)
        
        tokens_info.append({
            'text': token.text,
            'start': token_start,
            'end': token_end,
            'idx': token.i
        })
    
    # Find overlapping tokens
    overlapping = []
    for tok_info in tokens_info:
        if not (end <= tok_info['start'] or start >= tok_info['end']):
            overlapping.append(tok_info)
    
    return tokens_info, overlapping


def validate_alignment(text, entities, nlp, original_values=None, original_offsets=None):
    """
    Validate that entities align with token boundaries using spaCy.
    Returns (aligned_entities, misaligned_entities, warnings).
    
    Args:
        text: The text to validate against
        entities: List of (start, end, label) tuples
        nlp: spaCy model
        original_values: Dict mapping (start, end, label) -> original entity text
        original_offsets: Dict mapping (entity_text, label) -> (original_start, original_end) from ENTITIES_LABEL
    """
    from spacy.training import offsets_to_biluo_tags
    
    if original_values is None:
        original_values = {}
    if original_offsets is None:
        original_offsets = {}
    
    doc = nlp.make_doc(text)
    aligned = []
    misaligned = []
    warnings_list = []
    
    for start, end, label in entities:
        # Use char_span with alignment_mode="strict"
        span = doc.char_span(start, end, label=label, alignment_mode="strict")
        
        if span is None:
            # Get detailed token information
            all_tokens, overlapping_tokens = get_token_details(text, start, end, doc)
            
            # Categorize the misalignment
            error_type = categorize_misalignment(text, start, end, doc)
            
            # Try to get BILUO tags to show what would happen
            biluo_tags = None
            biluo_error = None
            try:
                biluo_tags = offsets_to_biluo_tags(doc, [(start, end, label)])
            except Exception as e:
                biluo_error = str(e)
            
            # Extract entity text safely
            entity_text = text[start:end] if start < len(text) and end <= len(text) else 'OUT_OF_BOUNDS'
            
            # Get expected value from original_values
            expected_value = original_values.get((start, end, label), 'UNKNOWN')
            
            # Get original offsets from ENTITIES_LABEL if available
            orig_start, orig_end = original_offsets.get((expected_value, label), (None, None))
            
            misaligned.append({
                'start': start,
                'end': end,
                'label': label,
                'entity_text': entity_text,
                'expected_value': expected_value,
                'original_char_start': orig_start,
                'original_char_end': orig_end,
                'error_type': error_type,
                'all_tokens': all_tokens,
                'overlapping_tokens': overlapping_tokens,
                'biluo_tags': biluo_tags,
                'biluo_error': biluo_error,
                'full_text': text
            })
            
            warnings_list.append(f"Misaligned entity '{entity_text}' ({label}) at [{start}:{end}]")
        else:
            aligned.append((start, end, label))
    
    return aligned, misaligned, warnings_list


def get_span_length_in_tokens(text, start, end, nlp):
    """Get the number of tokens in a span."""
    doc = nlp.make_doc(text)
    span = doc.char_span(start, end)
    if span is None:
        return None
    return len(span)


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_data(input_file, output_file):
    """Main function to process data and create spaCy training format."""
    
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load data
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df):,} rows from {input_file}")
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # Preprocess TEXT column
    print("\n" + "="*80)
    print("PREPROCESSING TEXT")
    print("="*80)
    print("Cleaning transaction text...")
    df['TEXT'] = df['TEXT'].apply(preprocess_text)
    print("✓ Text preprocessing complete")
    
    # Initialize spaCy
    print("\n" + "="*80)
    print("INITIALIZING SPACY")
    print("="*80)
    nlp = spacy.blank("en")
    print("✓ Loaded blank English model")
    
    # ========================================================================
    # PROCESS DATA
    # ========================================================================
    
    print("\n" + "="*80)
    print("PROCESSING TRANSACTIONS")
    print("="*80)
    
    # Convert dataframe to list of dicts (MUCH faster than iterrows)
    print(f"Using {cpu_count()} CPU cores for parallel processing...")
    print("Converting data to optimized format...")
    
    df_dicts = df.to_dict('records')
    
    # Add index and text column info to each dict
    for idx, row_dict in enumerate(df_dicts):
        row_dict['_idx'] = idx
        row_dict['_text_col'] = 'TEXT'
    
    results = []
    stats = {
        'total_rows': len(df),
        'empty_text': 0,
        'has_description_no_text': 0,
        'total_entities': 0,
        'aligned_entities': 0,
        'misaligned_entities': 0,
        'entity_counts': Counter(),
        'span_lengths': defaultdict(list),
        'misalignment_details': [],
        'misalignment_by_type': Counter(),
        'misalignment_by_label': Counter(),
        'rows_with_misalignment': set(),
        'rows_with_only_misalignment': set()
    }
    
    # Process in parallel with initializer (loads spaCy once per worker)
    with Pool(processes=cpu_count(), initializer=init_worker) as pool:
        for idx, result, row_stats in tqdm(
            pool.imap_unordered(process_row, df_dicts, chunksize=100),
            total=len(df_dicts),
            desc="Processing"
        ):
            # Check for empty text
            if result is None and row_stats['total_entities'] == 0:
                text = df.iloc[idx].get('TEXT', '')
                description = df.iloc[idx].get('DESCRIPTION', '')
                if pd.isna(text) or text == '':
                    stats['empty_text'] += 1
                    if not pd.isna(description) and description != '':
                        stats['has_description_no_text'] += 1
                continue
            
            # Aggregate stats
            stats['total_entities'] += row_stats['total_entities']
            stats['aligned_entities'] += row_stats['aligned_entities']
            stats['misaligned_entities'] += row_stats['misaligned_entities']
            
            for label, count in row_stats['entity_counts'].items():
                stats['entity_counts'][label] += count
            
            for label, lengths in row_stats['span_lengths'].items():
                stats['span_lengths'][label].extend(lengths)
            
            for error_type, count in row_stats['misalignment_by_type'].items():
                stats['misalignment_by_type'][error_type] += count
            
            for label, count in row_stats['misalignment_by_label'].items():
                stats['misalignment_by_label'][label] += count
            
            if row_stats['has_misalignment']:
                stats['rows_with_misalignment'].add(idx)
                if row_stats['only_misalignment']:
                    stats['rows_with_only_misalignment'].add(idx)
            
            # Store misalignment details
            stats['misalignment_details'].extend(row_stats['misalignment_details'])
            
            # Add result if available
            if result is not None:
                results.append(result)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(results_df):,} training examples to {output_file}")
    
    # ========================================================================
    # EDA ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS")
    print("="*80)
    
    # 1. Overall Statistics
    print("\n1. OVERALL STATISTICS")
    print("-" * 80)
    print(f"Total transactions: {stats['total_rows']:,}")
    print(f"Empty text: {stats['empty_text']:,} ({stats['empty_text']/stats['total_rows']*100:.2f}%)")
    print(f"Has DESCRIPTION but no TEXT: {stats['has_description_no_text']:,}")
    print(f"✓ Clean training data: {len(results_df):,} ({len(results_df)/stats['total_rows']*100:.2f}%)")
    
    print(f"\nEntity Extraction:")
    print(f"  Total entities found: {stats['total_entities']:,}")
    print(f"  ✓ Properly aligned: {stats['aligned_entities']:,} ({stats['aligned_entities']/stats['total_entities']*100:.2f}%)")
    print(f"  ✗ Alignment issues: {stats['misaligned_entities']:,} ({stats['misaligned_entities']/stats['total_entities']*100:.2f}%)")
    
    print(f"\nData Impact:")
    print(f"  Transactions with some issues: {len(stats['rows_with_misalignment']):,} ({len(stats['rows_with_misalignment'])/stats['total_rows']*100:.2f}%)")
    print(f"  Transactions completely lost: {len(stats['rows_with_only_misalignment']):,} ({len(stats['rows_with_only_misalignment'])/stats['total_rows']*100:.2f}%)")
    print(f"  ✓ Usable for training: {len(results_df):,} ({len(results_df)/stats['total_rows']*100:.2f}%)")
    
    # 2. Entities per Label
    print("\n2. SUCCESSFULLY EXTRACTED ENTITIES BY TYPE")
    print("-" * 80)
    print(f"{'Entity Type':<30} {'Count':<12} {'Percentage':<10}")
    print("-" * 80)
    
    total_aligned = sum(stats['entity_counts'].values())
    for label, count in sorted(stats['entity_counts'].items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_aligned * 100) if total_aligned > 0 else 0
        print(f"{label:<30} {count:<12,} {pct:<10.2f}%")
    
    # 3. Span Length Distribution
    print("\n3. ENTITY LENGTH DISTRIBUTION (in tokens)")
    print("-" * 80)
    print(f"{'Entity Type':<30} {'Min':<8} {'Avg':<8} {'Median':<8} {'Max':<8}")
    print("-" * 80)
    
    for label in sorted(stats['span_lengths'].keys()):
        lengths = stats['span_lengths'][label]
        if len(lengths) > 0:
            print(f"{label:<30} {min(lengths):<8} {np.mean(lengths):<8.2f} {np.median(lengths):<8.1f} {max(lengths):<8}")
    
    # 4. Issues by Type
    print("\n4. ALIGNMENT ISSUES BY TYPE")
    print("-" * 80)
    
    if stats['misaligned_entities'] > 0:
        print(f"{'Issue Type':<40} {'Count':<12} {'Percentage':<10}")
        print("-" * 80)
        for error_type, count in sorted(stats['misalignment_by_type'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / stats['misaligned_entities'] * 100)
            print(f"{error_type:<40} {count:<12,} {pct:<10.2f}%")
    else:
        print("✓ No alignment issues found!")
    
    # 5. Issues by Entity Type
    print("\n5. ALIGNMENT ISSUES BY ENTITY TYPE")
    print("-" * 80)
    
    if stats['misaligned_entities'] > 0:
        print(f"{'Entity Type':<30} {'Issues':<12} {'% of All Issues':<15}")
        print("-" * 80)
        for label, count in sorted(stats['misalignment_by_label'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / stats['misaligned_entities'] * 100)
            print(f"{label:<30} {count:<12,} {pct:<15.2f}%")
    
    # 6. Sample Issues by Type (2 per type, rest in CSV)
    print("\n6. SAMPLE ALIGNMENT ISSUES (2 per type)")
    print("="*80)
    
    if len(stats['misalignment_details']) > 0:
        # Group by error type
        issues_by_type = defaultdict(list)
        for detail in stats['misalignment_details']:
            issues_by_type[detail['error_type']].append(detail)
        
        # Prepare issues CSV data
        issues_csv_data = []
        
        for error_type, issues in sorted(issues_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n{'='*80}")
            print(f"ISSUE TYPE: {error_type}")
            print(f"Total occurrences: {len(issues):,}")
            print(f"{'='*80}")
            
            # Show 2 examples
            for i, entity in enumerate(issues[:2], 1):
                print(f"\nExample {i}:")
                print(f"  Transaction: {entity['full_text'][:80]}...")
                print(f"  Found in text: '{entity['entity_text']}'")
                print(f"  Expected value: '{entity['expected_value']}'")
                print(f"  Entity type: {entity['label']}")
                print(f"  Position: [{entity['start']}:{entity['end']}]")
                
                print(f"\n  Problem:")
                if 'start' in entity['error_type'].lower():
                    first_tok = entity['overlapping_tokens'][0] if entity['overlapping_tokens'] else None
                    if first_tok:
                        print(f"    Entity starts at position {entity['start']}, but this cuts into token '{first_tok['text']}'")
                        print(f"    Token '{first_tok['text']}' is at positions [{first_tok['start']}:{first_tok['end']}]")
                        print(f"    Should start at position {first_tok['start']} or {first_tok['end']}")
                elif 'end' in entity['error_type'].lower():
                    last_tok = entity['overlapping_tokens'][-1] if entity['overlapping_tokens'] else None
                    if last_tok:
                        print(f"    Entity ends at position {entity['end']}, but this cuts into token '{last_tok['text']}'")
                        print(f"    Token '{last_tok['text']}' is at positions [{last_tok['start']}:{last_tok['end']}]")
                        print(f"    Should end at position {last_tok['start']} or {last_tok['end']}")
                else:
                    print(f"    {entity['error_type']}")
                
                print(f"\n  Tokens in this part of text:")
                for tok in entity['overlapping_tokens'][:5]:  # Show first 5 tokens
                    print(f"    [{tok['start']:>3}:{tok['end']:<3}] '{tok['text']}'")
            
            if len(issues) > 2:
                print(f"\n  ... and {len(issues)-2:,} more examples (see issues.csv)")
            
            # Add all issues to CSV data
            for entity in issues:
                # Get tokenization for this text
                doc = nlp.make_doc(entity['full_text'])
                tokenized = [token.text for token in doc]
                
                issues_csv_data.append({
                    'row_index': entity.get('row_idx', 'unknown'),
                    'transaction_text': entity['full_text'],
                    'tokenized_text': ' | '.join(tokenized),
                    'issue_type': entity['error_type'],
                    'entity_label': entity['label'],
                    'found_in_text': entity['entity_text'],
                    'expected_value': entity['expected_value'],
                    'char_start': entity['start'],
                    'char_end': entity['end'],
                    'original_char_start': entity.get('original_char_start'),
                    'original_char_end': entity.get('original_char_end'),
                    'overlapping_tokens': ' | '.join([f"[{t['start']}:{t['end']}]='{t['text']}'" for t in entity['overlapping_tokens'][:10]])
                })
        
        # Save issues to CSV
        if issues_csv_data:
            issues_df = pd.DataFrame(issues_csv_data)
            issues_file = output_file.replace('.csv', '_issues.csv')
            issues_df.to_csv(issues_file, index=False)
            print(f"\n{'='*80}")
            print(f"✓ Saved all {len(issues_csv_data):,} issues to {issues_file}")
            print(f"{'='*80}")
    else:
        print("\n✓ No alignment issues found!")
    
    # 7. Example Clean Training Samples
    print("\n\n7. SAMPLE CLEAN TRAINING DATA")
    print("-" * 80)
    
    for i in range(min(5, len(results_df))):
        sample = results_df.iloc[i]
        entities = json.loads(sample['entities'])
        print(f"\nExample {i+1}:")
        print(f"  Transaction: {sample['text'][:100]}...")
        print(f"  Entities ({len(entities)}):")
        for start, end, label in entities:
            print(f"    - {label}: '{sample['text'][start:end]}' at [{start}:{end}]")
    
    # ========================================================================
    # SAVE STATISTICS
    # ========================================================================
    
    stats_file = output_file.replace('.csv', '_statistics.json')
    
    stats_export = {
        'total_rows': stats['total_rows'],
        'empty_text': stats['empty_text'],
        'has_description_no_text': stats['has_description_no_text'],
        'valid_training_examples': len(results_df),
        'total_entities': stats['total_entities'],
        'aligned_entities': stats['aligned_entities'],
        'misaligned_entities': stats['misaligned_entities'],
        'rows_with_misalignment': len(stats['rows_with_misalignment']),
        'rows_with_only_misalignment': len(stats['rows_with_only_misalignment']),
        'data_loss_pct': (len(stats['rows_with_only_misalignment']) / stats['total_rows'] * 100) if stats['total_rows'] > 0 else 0,
        'entity_counts': dict(stats['entity_counts']),
        'misalignment_by_type': dict(stats['misalignment_by_type']),
        'misalignment_by_label': dict(stats['misalignment_by_label']),
        'span_length_stats': {
            label: {
                'min': int(min(lengths)),
                'max': int(max(lengths)),
                'mean': float(np.mean(lengths)),
                'median': float(np.median(lengths))
            }
            for label, lengths in stats['span_lengths'].items() if len(lengths) > 0
        }
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats_export, f, indent=2)
    
    print(f"\n✓ Saved statistics to {stats_file}")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  1. Training data: {output_file} ({len(results_df):,} clean examples)")
    print(f"  2. Statistics: {stats_file}")
    if stats['misaligned_entities'] > 0:
        issues_file = output_file.replace('.csv', '_issues.csv')
        print(f"  3. Alignment issues: {issues_file} ({stats['misaligned_entities']:,} issues)")
    
    return results_df, stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results_df, stats = process_data(INPUT_FILE, OUTPUT_FILE)
