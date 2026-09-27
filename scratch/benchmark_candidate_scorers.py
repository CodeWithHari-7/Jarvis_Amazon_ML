import os
import sys
import re
import time
import unidecode
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import lightgbm as lgb
from collections import defaultdict

print("=" * 75)
print("STAGE 4: CANDIDATE SCORER BENCHMARKING (a -> b -> c -> d)")
print("=" * 75)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. LOAD & NORMALIZE STRATIFIED BENCHMARK SAMPLE
LEGAL_SUFFIXES = {'inc', 'corporation', 'corp', 'llc', 'limited', 'ltd', 'private', 'pvt', 'co', 'company', 'llp', 'pllc', 'sa', 'sarl', 'sas', 'sasu', 'eurl', 'gmbh'}

def normalize_record(name, addr, country):
    name_str = unidecode.unidecode(str(name or '')).lower()
    name_str = re.sub(r'&', ' and ', name_str)
    name_str = re.sub(r'\bcorp\b', 'corporation', name_str)
    name_str = re.sub(r'\bltd\b', 'limited', name_str)
    name_str = re.sub(r'\bpvt\b', 'private', name_str)
    name_str = re.sub(r'\brd\b', 'road', name_str)
    name_str = re.sub(r'\bst\b', 'street', name_str)
    name_clean = re.sub(r'[^a-z0-9\s]', ' ', name_str)
    name_tokens = [w for w in name_clean.split() if w]
    core_tokens = [w for w in name_tokens if w not in LEGAL_SUFFIXES]
    core_name = ' '.join(core_tokens) if core_tokens else ' '.join(name_tokens)
    
    addr_str = unidecode.unidecode(str(addr or '')).lower()
    addr_str = re.sub(r'&', ' and ', addr_str)
    addr_str = re.sub(r'\brd\b', 'road', addr_str)
    addr_str = re.sub(r'\bst\b', 'street', addr_str)
    addr_clean = re.sub(r'[^a-z0-9\s]', ' ', addr_str)
    addr_clean = re.sub(r'\s+', ' ', addr_clean).strip()
    
    nums = [n for n in re.findall(r'\d+', addr_clean) if len(n) <= 8]
    
    return {
        'country': str(country or '').strip().lower(),
        'raw_name': str(name or ''),
        'raw_addr': str(addr or ''),
        'norm_name': ' '.join(name_tokens),
        'core_name': core_name,
        'prefix4': core_name[:4] if len(core_name) >= 3 else core_name,
        'sorted_tok': ' '.join(sorted(core_tokens[:4])),
        'core_tokens': set(core_tokens),
        'norm_addr': addr_clean,
        'nums': set(nums)
    }

print("\n[1] Sampling Ground Truth (4,000 train S1, 1,000 holdout S1)...")
gt = pl.read_csv('dataset/train/train_ground_truth.tsv', separator='\t')

# Stratified sampling: 94.4% matched, 5.6% singletons
n_matched_total = 4720
n_singletons_total = 280

matched_gt = gt.filter(pl.col('matched_entity_ids').is_not_null() & (pl.col('matched_entity_ids') != '')).sample(n_matched_total, seed=42)
singleton_gt = gt.filter(pl.col('matched_entity_ids').is_null() | (pl.col('matched_entity_ids') == '')).sample(n_singletons_total, seed=42)
sample_gt = pl.concat([matched_gt, singleton_gt]).sample(fraction=1.0, shuffle=True, seed=42)

s1_ids = sample_gt['source1_entity_id'].to_list()
train_s1_ids = set(s1_ids[:4000])
val_s1_ids = set(s1_ids[4000:])

gt_dict = {}
all_target_ids = set()
for r in sample_gt.to_dicts():
    m = r['matched_entity_ids']
    t_set = set(m.split(',')) if m else set()
    gt_dict[r['source1_entity_id']] = t_set
    all_target_ids.update(t_set)

print(f"  Train S1: {len(train_s1_ids):,} | Val S1: {len(val_s1_ids):,}")
print(f"  Target S2/S3 IDs: {len(all_target_ids):,}")

# Load records
s1_df = pl.read_csv('dataset/train/train_source1.tsv', separator='\t').filter(pl.col('entity_id').is_in(set(s1_ids)))
s1_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s1_df.to_dicts()}

# Load candidate universe: all target S23 + 40k random S23 negatives per source
s2_full = pl.read_csv('dataset/train/train_source2.tsv', separator='\t')
s2_targets = s2_full.filter(pl.col('entity_id').is_in(all_target_ids))
s2_noise = s2_full.filter(~pl.col('entity_id').is_in(all_target_ids)).head(40000)

s3_full = pl.read_csv('dataset/train/train_source3.tsv', separator='\t')
s3_targets = s3_full.filter(pl.col('entity_id').is_in(all_target_ids))
s3_noise = s3_full.filter(~pl.col('entity_id').is_in(all_target_ids)).head(40000)

s23_df = pl.concat([s2_targets, s2_noise, s3_targets, s3_noise]).unique(subset=['entity_id'])
s23_records = {r['entity_id']: normalize_record(r['business_name'], r['business_address'], r['country']) for r in s23_df.to_dicts()}
print(f"  S23 Candidates pool: {len(s23_records):,}")

# Build Blocking Indices
idx_prefix = defaultdict(list)
idx_addr_num = defaultdict(list)
idx_sorted_tok = defaultdict(list)
idx_distinct_tok = defaultdict(list)

for eid, d in s23_records.items():
    c = d['country']
    if d['prefix4']: idx_prefix[(c, d['prefix4'])].append(eid)
    for n in d['nums']: idx_addr_num[(c, n)].append(eid)
    if d['sorted_tok']: idx_sorted_tok[(c, d['sorted_tok'])].append(eid)
    for tok in d['core_tokens']:
        if len(tok) >= 4: idx_distinct_tok[(c, tok)].append(eid)

def get_candidates(s1_id):
    d = s1_records[s1_id]
    c = d['country']
    cands = set(idx_prefix.get((c, d['prefix4']), []))
    for n in d['nums']: cands.update(idx_addr_num.get((c, n), []))
    cands.update(idx_sorted_tok.get((c, d['sorted_tok']), []))
    for tok in d['core_tokens']:
        if len(tok) >= 4:
            m = idx_distinct_tok.get((c, tok), [])
            if len(m) < 150: cands.update(m)
    return list(cands)

print("\n[2] Generating pairs for train and holdout...")
train_pairs = []
val_pairs = []

for s1_id in s1_ids:
    cands = get_candidates(s1_id)
    true_set = gt_dict[s1_id]
    is_train = s1_id in train_s1_ids
    
    # In training, ensure all true positives are included even if blocking missed, plus hard negatives
    for true_id in true_set:
        if true_id in s23_records and true_id not in cands:
            cands.append(true_id)
            
    for cand_id in cands:
        label = 1 if cand_id in true_set else 0
        pair = (s1_id, cand_id, label)
        if is_train:
            train_pairs.append(pair)
        else:
            val_pairs.append(pair)

train_df = pd.DataFrame(train_pairs, columns=['s1_id', 's23_id', 'label'])
val_df = pd.DataFrame(val_pairs, columns=['s1_id', 's23_id', 'label'])

print(f"  Train pairs: {len(train_df):,} (Pos: {train_df['label'].sum():,}, Neg: {(train_df['label']==0).sum():,})")
print(f"  Val pairs  : {len(val_df):,} (Pos: {val_df['label'].sum():,}, Neg: {(val_df['label']==0).sum():,})")

# Feature extraction function for metadata & string metrics
def compute_string_features(df):
    f_name_fuzz = []
    f_name_tok = []
    f_name_jw = []
    f_addr_fuzz = []
    f_addr_tok = []
    f_num_overlap = []
    
    for s1_id, s23_id in zip(df['s1_id'], df['s23_id']):
        d1 = s1_records[s1_id]
        d2 = s23_records[s23_id]
        
        n1, n2 = d1['norm_name'], d2['norm_name']
        a1, a2 = d1['norm_addr'], d2['norm_addr']
        
        f_name_fuzz.append(fuzz.ratio(n1, n2) / 100.0)
        f_name_tok.append(fuzz.token_set_ratio(n1, n2) / 100.0)
        f_name_jw.append(JaroWinkler.similarity(n1, n2))
        
        f_addr_fuzz.append(fuzz.ratio(a1, a2) / 100.0)
        f_addr_tok.append(fuzz.token_set_ratio(a1, a2) / 100.0)
        
        num1, num2 = d1['nums'], d2['nums']
        if num1 and num2:
            overlap = len(num1 & num2) / len(num1 | num2)
        elif not num1 and not num2:
            overlap = 1.0
        else:
            overlap = 0.0
        f_num_overlap.append(overlap)
        
    feats = pd.DataFrame({
        'name_fuzz': f_name_fuzz,
        'name_tok_set': f_name_tok,
        'name_jw': f_name_jw,
        'addr_fuzz': f_addr_fuzz,
        'addr_tok_set': f_addr_tok,
        'addr_num_overlap': f_num_overlap
    })
    return feats

print("\n[3] Extracting string features...")
X_train_str = compute_string_features(train_df)
X_val_str = compute_string_features(val_df)

# Official Macro F0.5 evaluation function
def evaluate_macro_f05(df, prob_col, threshold, val_s1_id_set, gt_mapping):
    scores = []
    pred_df = df[df[prob_col] >= threshold]
    pred_map = pred_df.groupby('s1_id')['s23_id'].apply(set).to_dict()
    
    for s1_id in val_s1_id_set:
        gt_set = gt_mapping.get(s1_id, set())
        pred_set = pred_map.get(s1_id, set())
        
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        # Singleton handling
        if len(gt_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
            continue
            
        if tp == 0:
            scores.append(0.0)
            continue
            
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        beta_sq = 0.25 # F0.5
        f05 = ((1 + beta_sq) * prec * rec) / (beta_sq * prec + rec)
        scores.append(f05)
        
    return np.mean(scores)

def find_best_f05(df, prob_col):
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.20, 0.96, 0.05):
        score = evaluate_macro_f05(df, prob_col, t, val_s1_ids, gt_dict)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score

# -------------------------------------------------------------
# (a) TF-IDF + Logistic Regression
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("EXPERIMENT (a): TF-IDF + Logistic Regression")
print("=" * 60)
t0 = time.time()
tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), max_features=10000)

train_texts_1 = [s1_records[s1]['norm_name'] + " " + s1_records[s1]['norm_addr'] for s1 in train_df['s1_id']]
train_texts_2 = [s23_records[s23]['norm_name'] + " " + s23_records[s23]['norm_addr'] for s23 in train_df['s23_id']]
val_texts_1 = [s1_records[s1]['norm_name'] + " " + s1_records[s1]['norm_addr'] for s1 in val_df['s1_id']]
val_texts_2 = [s23_records[s23]['norm_name'] + " " + s23_records[s23]['norm_addr'] for s23 in val_df['s23_id']]

tfidf.fit(train_texts_1[:5000] + train_texts_2[:5000])
# Compute cosine similarity as direct feature
train_vec1 = tfidf.transform(train_texts_1)
train_vec2 = tfidf.transform(train_texts_2)
val_vec1 = tfidf.transform(val_texts_1)
val_vec2 = tfidf.transform(val_texts_2)

tfidf_sim_train = np.asarray(train_vec1.multiply(train_vec2).sum(axis=1)).ravel()
tfidf_sim_val = np.asarray(val_vec1.multiply(val_vec2).sum(axis=1)).ravel()

lr_model = LogisticRegression(max_iter=300)
lr_model.fit(tfidf_sim_train.reshape(-1, 1), train_df['label'])
val_df['prob_tfidf_lr'] = lr_model.predict_proba(tfidf_sim_val.reshape(-1, 1))[:, 1]

t_lr, score_lr = find_best_f05(val_df, 'prob_tfidf_lr')
print(f"  [Model a] TF-IDF + Logistic Regression: Best Thresh = {t_lr:.2f} | Holdout F0.5 = {score_lr:.4f} | Time = {time.time()-t0:.1f}s")

# -------------------------------------------------------------
# (b) Char/Word Embedding + GRU (PyTorch)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("EXPERIMENT (b): Char Embedding + Bidirectional GRU")
print("=" * 60)
t0 = time.time()

CHAR_VOCAB = {c: i+1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 ")}

def text_to_tensor(text, max_len=64):
    indices = [CHAR_VOCAB.get(c, 0) for c in text[:max_len]]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices

class SeqDataset(Dataset):
    def __init__(self, df):
        self.s1 = [text_to_tensor(s1_records[s1]['norm_name']) for s1 in df['s1_id']]
        self.s2 = [text_to_tensor(s23_records[s23]['norm_name']) for s23 in df['s23_id']]
        self.labels = df['label'].values.astype(np.float32)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return (torch.tensor(self.s1[idx], dtype=torch.long),
                torch.tensor(self.s2[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32))

class SiameseGRU(nn.Module):
    def __init__(self, vocab_size=len(CHAR_VOCAB)+1, emb_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x1, x2):
        emb1 = self.embedding(x1)
        emb2 = self.embedding(x2)
        _, h1 = self.gru(emb1)
        _, h2 = self.gru(emb2)
        h1 = torch.cat([h1[0], h1[1]], dim=1)
        h2 = torch.cat([h2[0], h2[1]], dim=1)
        diff = torch.abs(h1 - h2)
        prod = h1 * h2
        feats = torch.cat([diff, prod], dim=1)
        return self.fc(feats).squeeze(-1)

train_loader = DataLoader(SeqDataset(train_df), batch_size=256, shuffle=True)
val_loader = DataLoader(SeqDataset(val_df), batch_size=512, shuffle=False)

gru_model = SiameseGRU().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(gru_model.parameters(), lr=1e-3)

gru_model.train()
for epoch in range(2):
    for x1, x2, y in train_loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        out = gru_model(x1, x2)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

gru_model.eval()
val_preds = []
with torch.no_grad():
    for x1, x2, _ in val_loader:
        x1, x2 = x1.to(device), x2.to(device)
        prob = torch.sigmoid(gru_model(x1, x2)).cpu().numpy()
        val_preds.extend(prob)

val_df['prob_gru'] = val_preds
t_gru, score_gru = find_best_f05(val_df, 'prob_gru')
print(f"  [Model b] Char Embedding + BiGRU: Best Thresh = {t_gru:.2f} | Holdout F0.5 = {score_gru:.4f} | Time = {time.time()-t0:.1f}s")

# -------------------------------------------------------------
# (c) MiniLM Sentence Embeddings + Cosine Similarity
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("EXPERIMENT (c): MiniLM Sentence Embeddings + Cosine (PyTorch CUDA)")
print("=" * 60)
t0 = time.time()
from transformers import AutoTokenizer, AutoModel

snapshot_dir = os.path.expanduser('~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41')
tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
minilm_model = AutoModel.from_pretrained(snapshot_dir, local_files_only=True).to(device)
minilm_model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_texts(texts, batch_size=128):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
        with torch.no_grad():
            output = minilm_model(**encoded)
            embs = mean_pooling(output, encoded['attention_mask'])
            embs = nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)

# Precompute embeddings for distinct entities in val
distinct_val_s1 = list(set(val_df['s1_id']))
distinct_val_s23 = list(set(val_df['s23_id']))

s1_texts = [s1_records[eid]['norm_name'] for eid in distinct_val_s1]
s23_texts = [s23_records[eid]['norm_name'] for eid in distinct_val_s23]

s1_embs = encode_texts(s1_texts)
s23_embs = encode_texts(s23_texts)

s1_emb_map = {eid: s1_embs[i] for i, eid in enumerate(distinct_val_s1)}
s23_emb_map = {eid: s23_embs[i] for i, eid in enumerate(distinct_val_s23)}

cos_sims = []
for s1, s23 in zip(val_df['s1_id'], val_df['s23_id']):
    e1 = s1_emb_map[s1]
    e2 = s23_emb_map[s23]
    cos_sims.append(float(torch.dot(e1, e2)))

val_df['prob_minilm'] = cos_sims
t_minilm, score_minilm = find_best_f05(val_df, 'prob_minilm')
print(f"  [Model c] MiniLM Sentence Embeddings + Cosine: Best Thresh = {t_minilm:.2f} | Holdout F0.5 = {score_minilm:.4f} | Time = {time.time()-t0:.1f}s")

# -------------------------------------------------------------
# (d) DistilBERT / Cross-Encoder Fine-Tuned Pair Classifier
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("EXPERIMENT (d): Cross-Encoder MiniLM / Pair Classifier (AMP on CUDA)")
print("=" * 60)
t0 = time.time()
# ms-marco-MiniLM cross-encoder is also cached locally!
cross_dir = os.path.expanduser('~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2/snapshots')
if os.path.exists(cross_dir):
    snap = os.listdir(cross_dir)[0]
    cross_path = os.path.join(cross_dir, snap)
    cross_tokenizer = AutoTokenizer.from_pretrained(cross_path, local_files_only=True)
    from transformers import AutoModelForSequenceClassification
    cross_model = AutoModelForSequenceClassification.from_pretrained(cross_path, local_files_only=True).to(device)
    cross_model.eval()
    
    cross_scores = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(val_df), batch_size):
            b_s1 = val_df['s1_id'].iloc[i:i+batch_size].tolist()
            b_s23 = val_df['s23_id'].iloc[i:i+batch_size].tolist()
            pairs = [[s1_records[s1]['norm_name'], s23_records[s2]['norm_name']] for s1, s2 in zip(b_s1, b_s23)]
            inputs = cross_tokenizer(pairs, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
            with torch.cuda.amp.autocast():
                logits = cross_model(**inputs).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            cross_scores.extend(probs)
    val_df['prob_cross'] = cross_scores
    t_cross, score_cross = find_best_f05(val_df, 'prob_cross')
    print(f"  [Model d] Cross-Encoder Classifier: Best Thresh = {t_cross:.2f} | Holdout F0.5 = {score_cross:.4f} | Time = {time.time()-t0:.1f}s")
else:
    score_cross = 0.0

# -------------------------------------------------------------
# META-CLASSIFIER: Top 2 Candidate Scores + String/Address Overlap -> LightGBM (GPU)
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("META-CLASSIFIER: Top Scores + String/Address Features -> LightGBM")
print("=" * 60)
t0 = time.time()

# Precompute MiniLM & TF-IDF scores on training set for Meta-Classifier
s1_train_distinct = list(set(train_df['s1_id']))
s23_train_distinct = list(set(train_df['s23_id']))
train_s1_embs = encode_texts([s1_records[eid]['norm_name'] for eid in s1_train_distinct])
train_s23_embs = encode_texts([s23_records[eid]['norm_name'] for eid in s23_train_distinct])
tr_s1_map = {eid: train_s1_embs[i] for i, eid in enumerate(s1_train_distinct)}
tr_s23_map = {eid: train_s23_embs[i] for i, eid in enumerate(s23_train_distinct)}

train_minilm_cos = [float(torch.dot(tr_s1_map[s1], tr_s23_map[s23])) for s1, s23 in zip(train_df['s1_id'], train_df['s23_id'])]

X_train_meta = X_train_str.copy()
X_train_meta['score_minilm'] = train_minilm_cos
X_train_meta['score_tfidf'] = tfidf_sim_train

X_val_meta = X_val_str.copy()
X_val_meta['score_minilm'] = val_df['prob_minilm']
X_val_meta['score_tfidf'] = tfidf_sim_val

lgb_train = lgb.Dataset(X_train_meta, label=train_df['label'])
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'device': 'cpu', # very fast for tabular tree
    'verbose': -1,
    'seed': 42
}
meta_lgb = lgb.train(params, lgb_train, num_boost_round=300)
val_df['prob_meta_lgb'] = meta_lgb.predict(X_val_meta)

t_meta, score_meta = find_best_f05(val_df, 'prob_meta_lgb')
print(f"  [Meta-Classifier LightGBM]: Best Thresh = {t_meta:.2f} | Holdout F0.5 = {score_meta:.4f} | Time = {time.time()-t0:.1f}s")

print("\n" + "=" * 75)
print("BENCHMARK SUMMARY ACROSS CANDIDATE SCORERS")
print("=" * 75)
print(f"  a. TF-IDF + Logistic Regression  : Holdout F0.5 = {score_lr:.4f}")
print(f"  b. Char Embedding + BiGRU        : Holdout F0.5 = {score_gru:.4f}")
print(f"  c. MiniLM Embeddings + Cosine    : Holdout F0.5 = {score_minilm:.4f}")
if score_cross > 0:
    print(f"  d. Cross-Encoder Pair Classifier : Holdout F0.5 = {score_cross:.4f}")
print(f"  --> Final Meta-Classifier (LGBM) : Holdout F0.5 = {score_meta:.4f}")
print("=" * 75)
