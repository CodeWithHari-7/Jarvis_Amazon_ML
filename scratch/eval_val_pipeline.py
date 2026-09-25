import sys, io, re, os, time, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import polars as pl
from collections import defaultdict
from indic_transliteration import sanscript

base_dir = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
cache_dir = os.path.join(base_dir, "pipeline_cache")

# Load subsets
s1_df = pd.read_csv(os.path.join(cache_dir, "subset_s1.tsv"), sep='\t', dtype=str).fillna("")
s2_df = pd.read_csv(os.path.join(cache_dir, "subset_s2.tsv"), sep='\t', dtype=str).fillna("")
s3_df = pd.read_csv(os.path.join(cache_dir, "subset_s3.tsv"), sep='\t', dtype=str).fillna("")
gt_df = pd.read_csv(os.path.join(cache_dir, "subset_gt.tsv"), sep='\t', dtype=str).fillna("")

sys.path.append(r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker")
from src.cleaning.data_cleaning import clean_dataset
from src.normalization.normalization import normalize_dataset
from src.features.feature_extraction import extract_features_for_pair
from src.models.inference import load_model, predict
from src.submission.format import format_submission

# 1. Normalize
s1_norm = normalize_dataset(clean_dataset(pl.from_pandas(s1_df))).to_dicts()
s2_norm = normalize_dataset(clean_dataset(pl.from_pandas(s2_df))).to_dicts()
s3_norm = normalize_dataset(clean_dataset(pl.from_pandas(s3_df))).to_dicts()
s23_norm = s2_norm + s3_norm

# 2. Transliterate
re_dev = re.compile(r'[\u0900-\u097F]')
re_guj = re.compile(r'[\u0A80-\u0AFF]')
re_tel = re.compile(r'[\u0C00-\u0C7F]')
re_clean = re.compile(r'[^a-z0-9\s]')

def translit_text(text: str) -> str:
    if not text: return ""
    s = str(text)
    try:
        if re_dev.search(s): s = sanscript.transliterate(s, sanscript.DEVANAGARI, sanscript.ITRANS)
        if re_guj.search(s): s = sanscript.transliterate(s, sanscript.GUJARATI, sanscript.ITRANS)
        if re_tel.search(s): s = sanscript.transliterate(s, sanscript.TELUGU, sanscript.ITRANS)
    except Exception: pass
    s = s.lower()
    s = re_clean.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

for r in s1_norm: r['norm_name_translit'] = translit_text(r['business_name_normalized'])
for r in s23_norm: r['norm_name_translit'] = translit_text(r['business_name_normalized'])

# 3. Blocking
prefix_idx = defaultdict(list)
addr_num_idx = defaultdict(list)
translit_idx = defaultdict(list)

for r in s23_norm:
    eid = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nt = r.get('norm_name_translit', '')
    nums = r.get('business_address_numbers', [])
    if c and n: prefix_idx[f"{c}_{n[:4]}"].append(eid)
    for num in nums:
        if c: addr_num_idx[f"{c}_{num}"].append(eid)
    if c and nt: translit_idx[f"{c}_{nt[:4]}"].append(eid)

candidates = set()
for r in s1_norm:
    s1_id = r['entity_id']
    c = r.get('country_normalized', '')
    n = r.get('business_name_normalized', '')
    nt = r.get('norm_name_translit', '')
    nums = r.get('business_address_numbers', [])
    if c and n:
        for m in prefix_idx.get(f"{c}_{n[:4]}", []): candidates.add((s1_id, m))
    for num in nums:
        if c:
            for m in addr_num_idx.get(f"{c}_{num}", []): candidates.add((s1_id, m))
    if c and nt:
        for m in translit_idx.get(f"{c}_{nt[:4]}", []): candidates.add((s1_id, m))

print(f"Total Unique Candidates: {len(candidates):,}")

# 4. Feature extraction
s1_lookup = {r['entity_id']: r for r in s1_norm}
s23_lookup = {r['entity_id']: r for r in s23_norm}

features = []
for s1_id, src_id in candidates:
    r1 = s1_lookup.get(s1_id)
    r2 = s23_lookup.get(src_id)
    if not r1 or not r2: continue
    feat = extract_features_for_pair(r1, r2)
    feat['source1_entity_id'] = s1_id
    feat['source_entity_id'] = src_id
    features.append(feat)

feat_df = pd.DataFrame(features)
print(f"Extracted features for {len(feat_df):,} candidates")

# 5. Model Inference
model = load_model(os.path.join(base_dir, "processed", "lgb_model.pkl"))
feat_cols = ['name_fuzz_ratio', 'name_token_set', 'name_jw', 
             'addr_fuzz_ratio', 'addr_token_set', 'addr_num_overlap', 'cross_script']
probs = predict(model, feat_df, feat_cols)
feat_df['prob'] = probs

# 6. Format Submission (threshold = 0.89)
s1_ids_list = s1_df['entity_id'].tolist()
sub_df = format_submission(s1_ids_list, feat_df['source1_entity_id'], feat_df['source_entity_id'], feat_df['prob'], threshold=0.89)

# 7. Evaluate Performance
merged = pd.merge(gt_df, sub_df, on='source1_entity_id', how='left', suffixes=('_gt', '_pred'))
merged['pred_list'] = merged['matched_entity_ids_pred'].apply(lambda x: set(str(x).split(',')) if x else set())
merged['gt_list'] = merged['matched_entity_ids_gt'].apply(lambda x: set(str(x).split(',')) if x else set())

total_tp = 0
total_fp = 0
total_fn = 0
f05_scores = []

for _, row in merged.iterrows():
    pred = row['pred_list']
    gt = row['gt_list']
    tp = len(pred.intersection(gt))
    fp = len(pred - gt)
    fn = len(gt - pred)
    total_tp += tp
    total_fp += fp
    total_fn += fn
    
    if len(gt) == 0:
        f05_scores.append(1.0 if len(pred) == 0 else 0.0)
    else:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        beta_sq = 0.25
        f05 = ((1 + beta_sq) * prec * rec / (beta_sq * prec + rec)) if (beta_sq * prec + rec) > 0 else 0.0
        f05_scores.append(f05)

macro_f05 = pd.Series(f05_scores).mean()
prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

print("\n=== VALIDATION METRICS (FINAL BLOCKING + FROZEN LGBM + THRESHOLD 0.89) ===")
print(f"True Positives (TP) : {total_tp:,}")
print(f"False Positives (FP): {total_fp:,}")
print(f"False Negatives (FN): {total_fn:,}")
print(f"Precision           : {prec*100:.2f}% ({prec:.4f})")
print(f"Recall              : {rec*100:.2f}% ({rec:.4f})")
print(f"Macro F0.5          : {macro_f05:.4f}")
