"""
ML Challenge 2026 — Business Entity Resolution
Module: predict.py

Loads serialized model artifact (GPU-accelerated XGBoost / LightGBM) and scores
candidate entity pairs using extracted RapidFuzz / character n-gram / numerical overlap features.
"""

import pickle
from typing import Dict, Any, List, Tuple
import numpy as np
from rapidfuzz import fuzz

from features import extract_pair_features, FEATURE_NAMES


class EntityMatcher:
    def __init__(
        self,
        model_artifact_path: str = "code/business_entity_resolution/src/model.pkl",
        threshold: float = 0.85
    ):
        with open(model_artifact_path, 'rb') as f:
            artifact = pickle.load(f)
        self.model = artifact['model']
        self.threshold = float(threshold) if threshold is not None else float(artifact.get('threshold', 0.85))
        self.feature_names = artifact.get('feature_names', FEATURE_NAMES)

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        """
        Computes continuous probability estimates [0.0, 1.0] across model types:
        - XGBClassifier: predict_proba(X)[:, 1]
        - XGBoost Booster: inplace_predict(X)
        - LightGBM: predict(X)
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        elif hasattr(self.model, 'inplace_predict'):
            return self.model.inplace_predict(X)
        else:
            return self.model.predict(X)

    def check_veto(self, rec1: Dict[str, Any], rec2: Dict[str, Any]) -> bool:
        """
        Hard veto rules (force no-match regardless of ML probability):
        1. Country mismatch
        2. Street number exact mismatch with low address similarity
        3. Core name token mismatch with low string similarity
        """
        # Rule 1: Country mismatch
        c1 = rec1.get('country', '')
        c2 = rec2.get('country', '')
        if c1 and c2 and c1 != c2:
            return True

        # Rule 2: Street number exact mismatch
        nums1 = rec1.get('nums', set())
        nums2 = rec2.get('nums', set())
        if nums1 and nums2 and not (nums1 & nums2):
            # Strict veto for France (avoids over-merging generic orgs on different street numbers)
            if c1 == 'france' or c2 == 'france':
                return True
            addr_sim = fuzz.token_set_ratio(rec1.get('norm_addr', ''), rec2.get('norm_addr', ''))
            if addr_sim < 75:
                return True

        # Rule 3: Core name token mismatch
        core1 = rec1.get('core_tokens', set())
        core2 = rec2.get('core_tokens', set())
        if core1 and core2 and not (core1 & core2):
            name_sim = fuzz.ratio(rec1.get('norm_name', ''), rec2.get('norm_name', ''))
            if name_sim < 65:
                return True

        return False

    def score_candidates(
        self,
        rec1: Dict[str, Any],
        candidate_records: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, float]]:
        """
        Scores candidate S2/S3 records against a reference S1 record.
        Returns list of (candidate_id, probability) pairs.
        """
        if not candidate_records:
            return []

        features_list = [extract_pair_features(rec1, rec2) for _, rec2 in candidate_records]
        X = np.array(features_list, dtype=np.float32)
        probs = self.predict_probs(X)

        return [(cand_id, float(prob)) for (cand_id, _), prob in zip(candidate_records, probs)]

    def predict_matches(
        self,
        rec1: Dict[str, Any],
        candidate_records: List[Tuple[str, Dict[str, Any]]],
        override_threshold: float = None,
        max_matches: int = 10
    ) -> List[str]:
        """
        Filters candidates by probability threshold, applies hard vetoes,
        and returns matched candidate IDs capped at max_matches (default 10, covering 99.98% of true clusters).
        """
        thresh = override_threshold if override_threshold is not None else self.threshold
        scored = self.score_candidates(rec1, candidate_records)

        rec_lookup = {cid: r for cid, r in candidate_records}
        valid_matches = []
        for cand_id, prob in scored:
            if prob >= thresh:
                rec2 = rec_lookup.get(cand_id, {})
                if not self.check_veto(rec1, rec2):
                    valid_matches.append((cand_id, prob))

        # Sort by confidence descending and cap at max_matches
        valid_matches.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in valid_matches[:max_matches]]
