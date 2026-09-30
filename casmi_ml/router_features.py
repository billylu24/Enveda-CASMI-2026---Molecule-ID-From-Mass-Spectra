"""Observable router features shared by evaluation and offline inference."""

import numpy as np

FEATURES = [
    "confidence",
    "margin",
    "relative_margin",
    "log_candidates",
    "query_spectra",
    "mean_peaks",
    "neutral_mass",
    "probability_entropy",
    "predicted_bits",
    "neural_score",
    "neural_gap",
    "reference_in_neural_rr",
    "neural_in_reference_rr",
    "neural_in_union_rr",
    "top1_agreement",
    "top5_overlap",
    "top25_overlap",
]


def rr(key, ranking):
    return 1 / (ranking.index(key) + 1) if key in ranking else 0.0


def router_features(group, probability, keys, fps, h, u, confidence, margin, center):
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    scores = fps @ (np.log(p) - np.log1p(-p)) / 2048
    order = sorted(range(len(keys)), key=lambda j: (-scores[j], keys[j]))
    neural = [keys[j] for j in order]
    histfirst = h[0] if h else None
    neuralfirst = neural[0] if neural else None
    return [
        confidence,
        margin,
        margin / max(confidence, 1e-6),
        np.log1p(len(keys)),
        len(group),
        float(np.mean([len(x) for x in group.ms2_mzs])),
        center or 0.0,
        float(-(p * np.log(p) + (1 - p) * np.log1p(-p)).mean()),
        float(p.sum()),
        float(scores[order[0]]) if order else -100.0,
        float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0,
        rr(histfirst, neural),
        rr(neuralfirst, h),
        rr(neuralfirst, u),
        float(histfirst is not None and histfirst == neuralfirst),
        len(set(h[:5]) & set(neural[:5])) / 5,
        len(set(h[:25]) & set(neural[:25])) / 25,
    ]
