"""Feature attribution for the stress probability.

For a logistic regression on standardised features, coef_i * z_i is the exact SHAP value of feature i
in log-odds space (interventional SHAP with the training mean as the baseline). The contributions add
up, with the intercept, to the model's logit, so every reading decomposes exactly into its drivers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .classifier import ClassifierResult
from .features import DRIVER_GROUPS, FEATURES


def attribute(features: pd.DataFrame, result: ClassifierResult) -> pd.DataFrame:
    """Per-date log-odds contributions by feature and by driver group, using the model that made each reading."""
    preds = result.predictions
    X = features.loc[preds.index]
    contrib = np.empty((len(X), X.shape[1]))
    base = np.empty(len(X))
    for model_id, idx in preds.groupby("model_id").indices.items():
        model = result.models[model_id]
        contrib[idx] = model.contributions(X.iloc[idx].to_numpy())
        base[idx] = model.intercept

    out = pd.DataFrame(contrib, index=X.index, columns=[f"c_{f}" for f in X.columns])
    for group in DRIVER_GROUPS:
        cols = [f"c_{f}" for f, (g, _) in FEATURES.items() if g == group]
        out[f"g_{group}"] = out[cols].sum(axis=1)
    out["base_logit"] = base
    out["logit"] = out[[f"c_{f}" for f in X.columns]].sum(axis=1) + base
    group_cols = [f"g_{g}" for g in DRIVER_GROUPS]
    out["top_driver"] = out[group_cols].idxmax(axis=1).str.removeprefix("g_")
    return out


def coefficient_table(result: ClassifierResult) -> pd.DataFrame:
    m = result.final_model
    return pd.DataFrame(
        {
            "feature": m.features,
            "driver_group": [FEATURES[f][0] for f in m.features],
            "std_coef": m.coef,
            "odds_ratio_per_1sd": np.exp(m.coef),
            "train_mean": m.mean,
            "train_sd": m.scale,
            "definition": [FEATURES[f][1] for f in m.features],
        }
    )
