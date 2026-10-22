"""Stress-regime ground truth built from documented market stress episodes."""

from __future__ import annotations

import pandas as pd


def episode_labels(index: pd.DatetimeIndex, episodes: list[list[str]]) -> pd.Series:
    """1 on dates inside any [start, end] episode (inclusive), else 0."""
    labels = pd.Series(0, index=index, name="stress", dtype=int)
    for start, end, *_ in episodes:
        labels.loc[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))] = 1
    return labels


def episode_name_series(index: pd.DatetimeIndex, episodes: list[list[str]]) -> pd.Series:
    names = pd.Series("", index=index, name="episode", dtype=object)
    for start, end, *rest in episodes:
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        names.loc[mask] = rest[0] if rest else f"{start}..{end}"
    return names
