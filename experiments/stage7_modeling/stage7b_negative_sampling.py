import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import os

base_dir   = r"c:\Users\PURUSOUTHANAN\OneDrive\Desktop\jarvis_checker\dataset"
proc_dir   = os.path.join(base_dir, "processed")
pq_in      = os.path.join(proc_dir, "ml_candidate_pairs.parquet")
pq_out     = os.path.join(proc_dir, "ml_train_candidates.parquet")

feat_cols = ['name_fuzz_ratio','name_token_set','name_jw',
             'addr_fuzz_ratio','addr_token_set','addr_num_overlap',
             'country_match','cross_script']

print("=== STAGE 7B — NEGATIVE SAMPLING REPORT ===\n")

# ── 1. LOAD CANDIDATE PAIRS ───────────────────────────────────
print("[1] Loading ml_candidate_pairs.parquet ...")
df = pd.read_parquet(pq_in)
pos_df = df[df['label'] == 1].copy()
neg_pool = df[df['label'] == 0].copy()
print(f"  Positives  : {len(pos_df):,}")
print(f"  Neg pool   : {len(neg_pool):,}")
print(f"  Ratio      : 1:{len(neg_pool)//len(pos_df)}")

# ── 2. NEGATIVE CATEGORY FUNCTIONS ───────────────────────────
# Category thresholds (chosen conservatively to avoid ambiguous labels)
HIGH_NAME  = 70   # name_token_set >= 70 → "similar name"
HIGH_ADDR  = 70   # addr_token_set  >= 70 → "similar address"

def tag_negative(row):
    ns  = row['name_token_set']
    as_ = row['addr_token_set']
    an  = row['addr_num_overlap']

    # Hardest: very high name + different address nums
    if ns >= 85 and an == 0.0:
        return "hard_similar_name_diff_addr"
    # High address, very different name
    if as_ >= 85 and ns < 40:
        return "hard_similar_addr_diff_name"
    # Both high — potential franchise / branch confusion
    if ns >= HIGH_NAME and as_ >= HIGH_ADDR:
        return "hard_both_similar"
    # Similar name only
    if ns >= HIGH_NAME:
        return "similar_name"
    # Similar address only
    if as_ >= HIGH_ADDR:
        return "similar_address"
    # Same country, low similarity — "easy" same-country negative
    if row['country_match'] == 1:
        return "same_country"
    # Random fallback (shouldn't occur much — blocking forces country match)
    return "random"

print("\n[2] Tagging negatives by type ...")
neg_pool['negative_type'] = neg_pool.apply(tag_negative, axis=1)
type_counts = neg_pool['negative_type'].value_counts()
print(type_counts.to_string())

# ── 3. SAMPLE STRATEGY ───────────────────────────────────────
# Target: 5× positives total negatives, with hard-negative enrichment
# Rationale: 1:5 ratio prevents trivially-easy training while preserving signal
N_POS       = len(pos_df)
TARGET_NEG  = N_POS * 5       # 5:1 overall
HARD_FRAC   = 0.50            # 50% of negatives must be "hard"
TARGET_HARD = int(TARGET_NEG * HARD_FRAC)
TARGET_EASY = TARGET_NEG - TARGET_HARD

hard_types = ['hard_similar_name_diff_addr',
              'hard_similar_addr_diff_name',
              'hard_both_similar',
              'similar_name',
              'similar_address']
easy_types = ['same_country', 'random']

hard_pool = neg_pool[neg_pool['negative_type'].isin(hard_types)]
easy_pool = neg_pool[neg_pool['negative_type'].isin(easy_types)]

# Sample hard negatives (cap per type to avoid imbalance within hard types)
hard_sampled = hard_pool.sample(min(TARGET_HARD, len(hard_pool)), random_state=42)
easy_sampled = easy_pool.sample(min(TARGET_EASY, len(easy_pool)), random_state=42)

print(f"\n[3] Negative sampling strategy (5:1 target, 50% hard)")
print(f"  Hard negatives available : {len(hard_pool):,}")
print(f"  Hard negatives sampled   : {len(hard_sampled):,}")
print(f"  Easy negatives sampled   : {len(easy_sampled):,}")

# ── 4. ASSEMBLE TRAINING SET ──────────────────────────────────
pos_df['negative_type'] = 'positive'
train_df = pd.concat([
    pos_df,
    hard_sampled,
    easy_sampled
], ignore_index=True)
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\n[4] Final training dataset")
print(f"  Positives  : {(train_df['label']==1).sum():,}")
print(f"  Negatives  : {(train_df['label']==0).sum():,}")
print(f"  Total      : {len(train_df):,}")
print(f"  Ratio      : 1:{(train_df['label']==0).sum() // max((train_df['label']==1).sum(),1)}")

neg_type_dist = train_df['negative_type'].value_counts()
print(f"\n  Negative type distribution:\n{neg_type_dist.to_string()}")

# ── 5. HARD NEGATIVE ANALYSIS ─────────────────────────────────
print("\n[5] Feature distributions by negative type")
for ntype in ['positive'] + hard_types:
    subset = train_df[train_df['negative_type'] == ntype]
    if len(subset) == 0:
        continue
    med = subset[feat_cols].median().round(2)
    print(f"\n  [{ntype}]  n={len(subset):,}")
    print(f"    name_token_set={med['name_token_set']}  addr_token_set={med['addr_token_set']}  addr_num_overlap={med['addr_num_overlap']}")

# ── 6. REAL EXAMPLES ──────────────────────────────────────────
print("\n[6] Representative hard-negative examples (top 5 per type)")
hard_show_types = ['hard_similar_name_diff_addr', 'hard_both_similar']
for ht in hard_show_types:
    subset = train_df[train_df['negative_type'] == ht].head(5)
    if len(subset) == 0:
        continue
    print(f"\n  [{ht}]")
    for _, r in subset.iterrows():
        print(f"    S1={r['source1_entity_id']}  Src={r['source_entity_id']} ({r['source']})")
        print(f"    name_ts={r['name_token_set']}  addr_ts={r['addr_token_set']}  num_ovlp={r['addr_num_overlap']}")

# ── 7. UNCERTAIN LABEL INVESTIGATION ─────────────────────────
print("\n[7] Uncertain label analysis")
# Any negative with name_token_set=100 AND addr_token_set=100 is suspicious
suspicious = train_df[(train_df['label']==0) & 
                       (train_df['name_token_set'] >= 95) & 
                       (train_df['addr_token_set'] >= 95)]
print(f"  Candidates labeled 0 with name_ts>=95 AND addr_ts>=95 : {len(suspicious):,}")
print(f"  RECOMMENDATION: Exclude these {len(suspicious):,} pairs from training (uncertain ground truth).")

# Drop uncertain pairs
train_df_clean = train_df.drop(
    train_df[(train_df['label']==0) &
             (train_df['name_token_set'] >= 95) &
             (train_df['addr_token_set'] >= 95)].index
)
print(f"  Training rows after exclusion: {len(train_df_clean):,}")

# ── 8. CLASS RATIO EXPERIMENTS ────────────────────────────────
print("\n[8] Class ratio options analysis")
for ratio in [1, 3, 5, 10]:
    n_neg = min(N_POS * ratio, len(neg_pool))
    print(f"  Ratio 1:{ratio} -> {N_POS:,} pos + {n_neg:,} neg = {N_POS + n_neg:,} total")

# ── 9. SAVE ───────────────────────────────────────────────────
cols_to_save = ['source1_entity_id','source_entity_id','source',
                'label','negative_type'] + feat_cols
train_df_clean[cols_to_save].to_parquet(pq_out, index=False)
print(f"\n[9] Saved: {pq_out}")
print(f"  Final shape: {train_df_clean[cols_to_save].shape}")

# ── 10. SUMMARY ───────────────────────────────────────────────
print("\n=== FINAL RECOMMENDATIONS ===")
print("  1. Use 1:5 pos:neg ratio — preserves statistical signal without drowning positives.")
print("  2. 50% of negatives must be hard — prevents model learning trivial easy-separations.")
print("  3. Exclude name_ts>=95 AND addr_ts>=95 negatives — they may be unlabeled true matches.")
print("  4. hard_similar_name_diff_addr is the most common dangerous pattern (franchise-like).")
print("  5. addr_num_overlap alone is sufficient to separate most hard negatives from positives.")
print("  6. negative_type column is metadata ONLY — do NOT include in ML feature vector.")
print("\n=== STAGE 7B COMPLETE ===")
