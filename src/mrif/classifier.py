"""Interpretable stress-regime classifier: standardised features + logistic regression.

Evaluated walk-forward: each calendar year from ``first_oos_year`` is scored by a model fitted only on
earlier data, so the reported ROC-AUC is out of sample. Every fitted model is kept as plain parameters
(mean, scale, coefficients, intercept) so any probability can be recomputed by hand during validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .config import ClassifierConfig


@dataclass
class FittedModel:
    model_id: str
    features: list[str]
    train_start: str
    train_end: str
    n_train: int
    n_stress: int
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float

    def standardise(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - np.asarray(self.mean)) / np.asarray(self.scale)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """Per-feature log-odds contributions; they sum with the intercept to the model logit."""
        return self.standardise(X) * np.asarray(self.coef)

    def logit(self, X: np.ndarray) -> np.ndarray:
        return self.contributions(X).sum(axis=-1) + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.logit(X)))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FittedModel":
        return cls(**d)


def fit_model(X: pd.DataFrame, y: pd.Series, cfg: ClassifierConfig, model_id: str) -> FittedModel:
    if y.nunique() < 2:
        raise ValueError(f"{model_id}: training data must contain both stable and stress days")
    scaler = StandardScaler().fit(X.to_numpy())
    clf = LogisticRegression(C=cfg.C, class_weight=cfg.class_weight, max_iter=1000)
    clf.fit(scaler.transform(X.to_numpy()), y.to_numpy())
    return FittedModel(
        model_id=model_id,
        features=list(X.columns),
        train_start=str(X.index[0].date()),
        train_end=str(X.index[-1].date()),
        n_train=len(X),
        n_stress=int(y.sum()),
        mean=scaler.mean_.tolist(),
        scale=scaler.scale_.tolist(),
        coef=clf.coef_[0].tolist(),
        intercept=float(clf.intercept_[0]),
    )


@dataclass
class ClassifierResult:
    predictions: pd.DataFrame        # date -> prob, label, model_id, in_sample
    models: dict[str, FittedModel]   # model_id -> parameters
    final_model: FittedModel         # fitted on the whole sample, for coefficient reporting
    metrics: dict


def walk_forward(features: pd.DataFrame, labels: pd.Series, cfg: ClassifierConfig) -> ClassifierResult:
    y = labels.reindex(features.index).fillna(0).astype(int)
    years = features.index.year
    oos_years = sorted(int(v) for v in np.unique(years) if v >= cfg.first_oos_year)
    if not oos_years:
        raise ValueError("no out-of-sample years: lower classifier.first_oos_year")

    models: dict[str, FittedModel] = {}
    parts = []
    for yr in oos_years:
        train = years < yr
        test = years == yr
        model = fit_model(features[train], y[train], cfg, model_id=f"wf-{yr}")
        models[model.model_id] = model
        parts.append(
            pd.DataFrame(
                {"prob": model.predict_proba(features[test].to_numpy()), "model_id": model.model_id},
                index=features.index[test],
            )
        )
    oos = pd.concat(parts)

    final = fit_model(features, y, cfg, model_id="full-sample")
    models[final.model_id] = final
    ins_idx = features.index[years < oos_years[0]]
    ins = pd.DataFrame({"prob": final.predict_proba(features.loc[ins_idx].to_numpy()), "model_id": final.model_id}, index=ins_idx)

    preds = pd.concat([ins.assign(in_sample=True), oos.assign(in_sample=False)]).sort_index()
    preds["label"] = y.reindex(preds.index)
    preds.index.name = "date"

    metrics = evaluate(oos["prob"], y.reindex(oos.index), cfg.threshold)
    metrics["in_sample_roc_auc_full_model"] = float(roc_auc_score(y, final.predict_proba(features.to_numpy())))
    metrics["oos_start"] = str(oos.index[0].date())
    metrics["oos_end"] = str(oos.index[-1].date())
    metrics["roc_auc_by_year"] = {
        str(yr): float(roc_auc_score(y[years == yr], oos.loc[features.index[years == yr], "prob"]))
        for yr in oos_years
        if y[years == yr].nunique() == 2
    }
    return ClassifierResult(predictions=preds, models=models, final_model=final, metrics=metrics)


def evaluate(prob: pd.Series, y: pd.Series, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "oos_roc_auc": float(roc_auc_score(y, prob)),
        "oos_brier": float(brier_score_loss(y, prob)),
        "oos_precision": float(precision_score(y, pred, zero_division=0)),
        "oos_recall": float(recall_score(y, pred, zero_division=0)),
        "oos_n": int(len(y)),
        "oos_stress_days": int(y.sum()),
        "threshold": threshold,
    }
