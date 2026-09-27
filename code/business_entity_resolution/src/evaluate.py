"""Official metric: macro F0.5 per Source 1 entity (singletons included) + blocking recall."""
import numpy as np


def macro_f05(ids, gt, pred):
    """ids: S1 ids evaluated; gt/pred: dict id -> set(ids). Returns (F0.5, micro P, micro R, singleton acc)."""
    sc, tp, fp, fn, s_tot, s_ok = [], 0, 0, 0, 0, 0
    for s in ids:
        t, q = gt.get(s, set()), pred.get(s, set())
        a = len(t & q)
        tp += a; fp += len(q) - a; fn += len(t) - a
        if not t:
            s_tot += 1; s_ok += not q
            sc.append(1.0 if not q else 0.0)
        elif a == 0:
            sc.append(0.0)
        else:
            P, R = a / len(q), a / len(t)
            sc.append(1.25 * P * R / (0.25 * P + R))
    return float(np.mean(sc)), tp / max(tp + fp, 1), tp / max(tp + fn, 1), s_ok / max(s_tot, 1)


def blocking_recall(ids, gt, cands):
    tot = sum(len(gt[s]) for s in ids)
    return sum(len(gt[s] & cands.get(s, set())) for s in ids) / max(tot, 1)
