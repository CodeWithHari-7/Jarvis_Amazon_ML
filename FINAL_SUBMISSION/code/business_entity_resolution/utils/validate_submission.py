"""
Official-style submission validator for JARVIS_CHECKER.
Mirrors the logic of utils/validate_submission.py from the challenge student_resource.

Usage:
    python utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"""
import argparse
import sys
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description='Validate submission files')
    parser.add_argument('--matching', required=True, help='Path to matching_results.tsv')
    parser.add_argument('--candidate', required=True, help='Path to candidate_pairs.tsv')
    parser.add_argument('--test-dir', required=True, help='Path to test directory')
    args = parser.parse_args()

    errors = []
    warnings_list = []

    print("=" * 70)
    print("SUBMISSION VALIDATOR")
    print("=" * 70)

    # ── Check files exist ─────────────────────────────────────────────────────
    for path in [args.matching, args.candidate]:
        if not os.path.exists(path):
            print(f"ERROR: File not found: {path}")
            sys.exit(1)

    test_s1_path = os.path.join(args.test_dir, 'test_source1.tsv')
    test_s2_path = os.path.join(args.test_dir, 'test_source2.tsv')
    test_s3_path = os.path.join(args.test_dir, 'test_source3.tsv')

    for path in [test_s1_path, test_s2_path, test_s3_path]:
        if not os.path.exists(path):
            print(f"ERROR: Test file not found: {path}")
            sys.exit(1)

    # ── Load test IDs ─────────────────────────────────────────────────────────
    print("\n[1] Loading test entity IDs...")
    s1_df = pd.read_csv(test_s1_path, sep='\t', dtype=str, usecols=['entity_id'],
                        keep_default_na=False)
    s2_df = pd.read_csv(test_s2_path, sep='\t', dtype=str, usecols=['entity_id'],
                        keep_default_na=False)
    s3_df = pd.read_csv(test_s3_path, sep='\t', dtype=str, usecols=['entity_id'],
                        keep_default_na=False)

    s1_ids = set(s1_df['entity_id'])
    s23_ids = set(s2_df['entity_id']) | set(s3_df['entity_id'])

    print(f"  Test S1: {len(s1_ids):,}")
    print(f"  Test S2+S3: {len(s23_ids):,}")

    # ── Load submission files ─────────────────────────────────────────────────
    print("\n[2] Loading submission files...")
    try:
        match_df = pd.read_csv(args.matching, sep='\t', dtype=str, keep_default_na=False)
        print(f"  matching_results.tsv: {len(match_df):,} rows")
    except Exception as e:
        errors.append(f"Cannot parse matching_results.tsv: {e}")
        _report_and_exit(errors)

    try:
        cand_df = pd.read_csv(args.candidate, sep='\t', dtype=str, keep_default_na=False)
        print(f"  candidate_pairs.tsv:  {len(cand_df):,} rows")
    except Exception as e:
        errors.append(f"Cannot parse candidate_pairs.tsv: {e}")
        _report_and_exit(errors)

    # ── Column name check ─────────────────────────────────────────────────────
    print("\n[3] Column name validation...")
    expected_match_cols = ['source1_entity_id', 'matched_entity_ids']
    expected_cand_cols = ['source1_entity_id', 'candidate_entity_ids']

    if list(match_df.columns) != expected_match_cols:
        errors.append(
            f"matching_results.tsv has wrong columns: {list(match_df.columns)}, "
            f"expected: {expected_match_cols}"
        )
    else:
        print("  PASS: matching_results.tsv columns correct")

    if list(cand_df.columns) != expected_cand_cols:
        errors.append(
            f"candidate_pairs.tsv has wrong columns: {list(cand_df.columns)}, "
            f"expected: {expected_cand_cols}"
        )
    else:
        print("  PASS: candidate_pairs.tsv columns correct")

    # ── S1 coverage in matching_results.tsv ──────────────────────────────────
    print("\n[4] S1 coverage check (matching_results.tsv)...")
    result_s1 = set(match_df['source1_entity_id'])
    missing_s1 = s1_ids - result_s1
    extra_s1 = result_s1 - s1_ids

    if len(match_df) == len(s1_ids) and not missing_s1 and not extra_s1:
        print(f"  PASS: All {len(s1_ids):,} S1 entities present, no extras")
    else:
        if missing_s1:
            errors.append(
                f"[matching_results] ERROR: {len(missing_s1):,} source1_entity_id(s) missing from submission."
            )
        if extra_s1:
            errors.append(
                f"[matching_results] ERROR: {len(extra_s1):,} source1_entity_id(s) in submission not in test set."
            )

    # ── No duplicate S1 IDs ───────────────────────────────────────────────────
    print("\n[5] Duplicate S1 ID check...")
    dup_match = len(match_df) - match_df['source1_entity_id'].nunique()
    dup_cand = len(cand_df) - cand_df['source1_entity_id'].nunique()

    if dup_match == 0:
        print("  PASS: No duplicate S1 IDs in matching_results.tsv")
    else:
        errors.append(f"matching_results.tsv: {dup_match} duplicate source1_entity_id rows")

    if dup_cand == 0:
        print("  PASS: No duplicate S1 IDs in candidate_pairs.tsv")
    else:
        errors.append(f"candidate_pairs.tsv: {dup_cand} duplicate source1_entity_id rows")

    # ── Matched ID validity ───────────────────────────────────────────────────
    print("\n[6] Matched entity ID validity...")
    invalid_ids = []
    s1_in_matched = []
    dup_in_row = []
    not_in_s23 = []

    for _, row in match_df.iterrows():
        s1_id = row['source1_entity_id']
        m_str = row['matched_entity_ids']
        if not m_str:
            continue
        m_list = [m.strip() for m in m_str.split(',') if m.strip()]
        m_set = set(m_list)
        if len(m_set) < len(m_list):
            dup_in_row.append(s1_id)
        for m in m_list:
            if m.startswith('S1-') or m.startswith('S1'):
                s1_in_matched.append((s1_id, m))
            if not (m.startswith('S2') or m.startswith('S3')):
                invalid_ids.append((s1_id, m))
            if m not in s23_ids:
                not_in_s23.append((s1_id, m))

    if not invalid_ids:
        print("  PASS: All matched IDs have valid S2/S3 prefix")
    else:
        errors.append(f"Found {len(invalid_ids)} matched IDs without S2/S3 prefix")

    if not s1_in_matched:
        print("  PASS: No S1 IDs in matched_entity_ids")
    else:
        errors.append(f"Found {len(s1_in_matched)} S1 IDs inside matched_entity_ids")

    if not dup_in_row:
        print("  PASS: No duplicate IDs within a single prediction row")
    else:
        errors.append(f"Found {len(dup_in_row)} rows with duplicate matched IDs")

    if not not_in_s23:
        print("  PASS: All matched IDs exist in test S2/S3")
    else:
        errors.append(f"Found {len(not_in_s23)} matched IDs not in test S2/S3")

    # ── Candidate ID validity ─────────────────────────────────────────────────
    print("\n[7] Candidate pairs validation...")
    # Build full candidate sets
    cand_lookup = {}
    for _, row in cand_df.iterrows():
        s1_id = row['source1_entity_id']
        c_str = row['candidate_entity_ids']
        cand_lookup[s1_id] = set(c_str.split(',')) if c_str else set()

    # Every final match must be in candidates
    match_not_in_cands = 0
    for _, row in match_df.iterrows():
        s1_id = row['source1_entity_id']
        m_str = row['matched_entity_ids']
        if not m_str:
            continue
        m_list = [m.strip() for m in m_str.split(',') if m.strip()]
        cands_for_s1 = cand_lookup.get(s1_id, set())
        for m in m_list:
            if m not in cands_for_s1:
                match_not_in_cands += 1

    if match_not_in_cands == 0:
        print("  PASS: All final matches are in candidate_pairs.tsv")
    else:
        errors.append(f"{match_not_in_cands} final matched IDs not present in candidate_pairs.tsv")

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if errors:
        print("VALIDATION: FAIL")
        print("=" * 70)
        for err in errors:
            print(f"  ERROR: {err}")
        sys.exit(1)
    else:
        print("VALIDATION: PASS")
        print("=" * 70)
        # Summary stats
        n_matched = match_df[match_df['matched_entity_ids'] != ''].shape[0]
        print(f"  S1 entities: {len(s1_ids):,}")
        print(f"  S1 with matches: {n_matched:,} ({100*n_matched/len(s1_ids):.1f}%)")
        print(f"  S1 with no match: {len(s1_ids)-n_matched:,} ({100*(len(s1_ids)-n_matched)/len(s1_ids):.1f}%)")
        sys.exit(0)


def _report_and_exit(errors):
    print("\nVALIDATION: FAIL")
    for err in errors:
        print(f"  ERROR: {err}")
    sys.exit(1)


if __name__ == '__main__':
    main()
