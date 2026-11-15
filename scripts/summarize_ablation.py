#!/usr/bin/env python
"""Paired ΔMAE, pooled trimmed means, and support/error correlations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data_io import group3, read_rankings


def trimmed_mean(values, fraction=0.1):
    if not 0 <= fraction < 0.5:
        raise ValueError('Trim fraction must be in [0,0.5)')
    values = np.sort(np.asarray(values, float))
    values = values[np.isfinite(values)]
    cut = int(len(values) * fraction)
    kept = values[cut:len(values) - cut]
    return (float(kept.mean()) if len(kept) else np.nan), len(kept)


def summarize(frame, min_count=100, trim=0.1, include_group3=False, scores=None, threshold=0.4):
    """Summarize f_X > threshold rows of held-out elements with count > min_count."""
    frame = frame.copy()
    if 'subset' in frame:
        frame = frame.loc[frame['subset'].eq('threshold') & np.isclose(frame['lower'], threshold)]
    if not include_group3:
        frame = frame.loc[~frame.element.map(group3)]
    frame = frame.loc[frame['count'].gt(min_count)].copy()
    keys = ['element', 'repeat']
    if frame.duplicated(keys + ['similarity']).any():
        raise ValueError('Duplicate element/repeat/condition metrics; summarize one run at a time')
    base = frame.loc[frame.similarity.eq('Random')].set_index(keys)
    index = pd.MultiIndex.from_frame(frame[keys])
    if not index.isin(base.index).all():
        raise ValueError('Missing paired baseline')
    paired = base.reindex(index)
    if not np.array_equal(frame['count'].to_numpy(), paired['count'].to_numpy()):
        raise ValueError('Condition and baseline have different test counts')
    if 'test_ids_sha256' in frame and not np.array_equal(frame.test_ids_sha256.to_numpy(), paired.test_ids_sha256.to_numpy()):
        raise ValueError('Condition and baseline have different test identifiers')
    frame['delta_MAE'] = frame.MAE.to_numpy() - paired.MAE.to_numpy()
    rows = []
    for condition, selected in frame.groupby('similarity', sort=False):
        value, retained = trimmed_mean(selected.delta_MAE, trim)
        rows.append(dict(similarity=condition, mean_delta_MAE=value, mean_delta_MAE_meV=value * 1000,
                         n_elements=selected.element.nunique(), n_observations=len(selected),
                         n_retained=retained, trim_each_tail=trim,
                         elements=' '.join(sorted(selected.element.unique()))))
    baseline = frame.loc[frame.similarity.eq('Random')].copy()
    if scores is not None:
        baseline['support_top6'] = baseline.element.map(lambda e: np.mean(scores[e][:6]) if e in scores and len(scores[e]) >= 6 else np.nan)
    if 'support_top6' not in baseline:
        return pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame()
    per = baseline.groupby('element')[['MAE', 'support_top6']].mean().reset_index()
    valid = per.loc[np.isfinite(per.MAE) & np.isfinite(per.support_top6)]
    rho = pvalue = np.nan
    if len(valid) >= 3 and valid.support_top6.nunique() > 1 and valid.MAE.nunique() > 1:
        rho, pvalue = spearmanr(valid.support_top6, valid.MAE)
    correlations = pd.DataFrame([dict(n_elements=len(valid), spearman_rho=rho, pvalue=pvalue)])
    return pd.DataFrame(rows), per, correlations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', type=Path, required=True, help='metrics.csv of a run, or a reference result CSV')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--support-csv', type=Path, help='ranking file for the top-six score correlation')
    p.add_argument('--min-count', type=int, default=100, help='require more than this many test compounds')
    p.add_argument('--trim', type=float, default=0.1, help='fraction removed from each tail')
    p.add_argument('--include-group3', action='store_true', help='keep group-3 held-out elements')
    args = p.parse_args()
    frame = pd.read_csv(args.metrics)
    frame['similarity'] = frame.similarity.replace({'10-15 excluded': 'Ranks 11-16 excluded'})
    scores = read_rankings(args.support_csv)[1] if args.support_csv else None
    summary, per, correlation = summarize(frame, args.min_count, args.trim, args.include_group3, scores)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError('Summary output directory must be empty')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / 'summary.csv', index=False)
    per.to_csv(args.out_dir / 'support_error.csv', index=False)
    correlation.to_csv(args.out_dir / 'correlations.csv', index=False)


if __name__ == '__main__':
    main()
