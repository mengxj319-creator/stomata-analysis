"""Comparisons between groups, with the sampling unit (e.g. plant) as the experimental unit."""

import itertools

import numpy as np
import pandas as pd
from scipy import stats


def group_pairs(groups, requested):
    """requested: list of [group_a, group_b] from the configuration; empty = all pairs."""
    if requested:
        missing = {g for pair in requested for g in pair} - set(groups)
        if missing:
            raise ValueError(f"compare_groups: unknown group(s) {sorted(missing)}; groups found: {sorted(groups)}")
        return [tuple(p) for p in requested]
    return list(itertools.combinations(sorted(groups), 2))


def compare_units(units, value_cols, pairs, by=()):
    """units: one row per unit with columns 'group', 'unit', value_cols and the stratifying columns `by`.
    Welch t-test and Mann-Whitney U for each pair of groups; pairs with < 2 units in a group are skipped."""
    rows = []
    strata = units.groupby(list(by)) if by else [((), units)]
    for key, d in strata:
        key = key if isinstance(key, tuple) else (key,)
        for a, b in pairs:
            for col in value_cols:
                x = d.loc[d.group == a, col].dropna().astype(float)
                y = d.loc[d.group == b, col].dropna().astype(float)
                row = dict(zip(by, key), variable=col, group_a=a, group_b=b, n_a=len(x), n_b=len(y),
                           mean_a=x.mean(), mean_b=y.mean(), welch_p=np.nan, mann_whitney_p=np.nan)
                if len(x) >= 2 and len(y) >= 2:
                    row["welch_p"] = stats.ttest_ind(x, y, equal_var=False).pvalue
                    row["mann_whitney_p"] = stats.mannwhitneyu(x, y, alternative="two-sided").pvalue
                rows.append(row)
    return pd.DataFrame(rows)
