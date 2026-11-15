"""Fixed-size leave-one-element-out source-exclusion experiment."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import (DATABASES, MEASURES, align, group3, load_database, load_features,
                     ranking_path, read_rankings, same_group, sha256, write_json)

CONDITIONS = ('Random', 'Top 2 excluded', 'Top 4 excluded', 'Top 6 excluded',
              'Ranks 11-16 excluded', 'Same group excluded')
THRESHOLD = 0.4


def exclusions(element, support, available):
    if len(support) < 16:
        raise ValueError('At least 16 ranked source elements are required')
    return dict(zip(CONDITIONS, [[], support[:2], support[:4], support[:6], support[10:16],
                                same_group(element, available)]))


def population_masks(database, element, excluded, filtered=True, threshold=THRESHOLD):
    train = database[f'ratio_{element}'].eq(0)
    test = database[f'ratio_{element}'].gt(threshold)
    if filtered:
        columns = [f'ratio_{e}' for e in excluded]
        missing = set(columns).difference(database.columns)
        if missing:
            raise ValueError(f'Missing source fraction columns: {sorted(missing)}')
        test &= ~database[columns].gt(0).any(axis=1)
    return train, test


def sample_pool(database, train_mask, removed, size, seed):
    pool = database.loc[train_mask]
    if removed:
        pool = pool.loc[~pool[[f'ratio_{e}' for e in removed]].gt(0).any(axis=1)]
    if len(pool) < size:
        raise ValueError(f'pool has {len(pool)} compounds; requires exactly {size}')
    return pool.sample(n=size, random_state=seed).index


def validate_ids(database, ids, element, removed, n_train):
    if len(ids) != n_train or not ids.is_unique or not ids.isin(database.index).all():
        raise ValueError('Training IDs have wrong size, duplicates, or unavailable compounds')
    selected = database.loc[ids]
    if selected[f'ratio_{element}'].gt(0).any():
        raise ValueError('Held-out element leaked into training')
    if removed and selected[[f'ratio_{e}' for e in removed]].gt(0).any(axis=None):
        raise ValueError('Excluded source leaked into training')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='folder downloaded from the data repository')
    p.add_argument('--out-dir', type=Path, required=True, help='new or empty output folder')
    p.add_argument('--db', choices=DATABASES, default='OQMD')
    p.add_argument('--task', choices=['e_form', 'bandgap'], default='e_form')
    p.add_argument('--representation', choices=['pcg', 'matminer'], default='pcg', help='regressor features')
    p.add_argument('--ranking', choices=MEASURES, default='pcg', help='source ranking measure')
    p.add_argument('--support-csv', type=Path, help='ranking file to use instead of the published one')
    p.add_argument('--method', choices=['RF', 'XGB', 'SVR'], default='SVR')
    p.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    p.add_argument('--gpu-id', type=int)
    p.add_argument('--elements', nargs='+', help='held-out elements (default: all ranked elements)')
    p.add_argument('--train-size', type=int, default=10000)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--seed', type=int, default=42, help='repetition r uses seed + r')
    p.add_argument('--min-test', type=int, default=100, help='require more than this many test compounds')
    p.add_argument('--test-mode', choices=['filtered', 'unfiltered'], default='filtered',
                   help='filtered: drop test compounds containing any of the top-6 sources')
    args = p.parse_args(argv)
    if min(args.train_size, args.repeats, args.min_test) < 1:
        p.error('train-size, repeats, and min-test must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    from regressors import build_model, metrics
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    support_path = args.support_csv or ranking_path(args.data_dir, args.ranking)
    rankings, scores = read_rankings(support_path)
    database = load_database(args.data_dir, args.db)
    database = database.loc[np.isfinite(database[args.task])]
    features, feature_report = load_features(args.data_dir, args.db, args.representation)
    database, features, alignment_report = align(database, features)
    available = [c[6:] for c in database if c.startswith('ratio_')]
    candidates = sorted(args.elements or rankings)
    config = {**vars(args), 'support_csv': str(support_path), 'support_sha256': sha256(support_path),
              'feature_report': feature_report, 'alignment': alignment_report,
              'test_population': f'f_X > {THRESHOLD}', 'conditions': list(CONDITIONS),
              'summary': 'group 3 excluded; count > min_test; 10% trimmed from each tail',
              'scaler': 'StandardScaler fitted on each sampled training set (SVR only)',
              'target_dtype': 'float32', 'feature_dtype': 'float32',
              'prediction_storage': dict(path='predictions/{element}.csv.gz',
                  columns=['element', 'material_id', 'repeat', 'condition', 'true', 'prediction', 'ratio_X'])}
    write_json(out / 'config.json', config)
    prediction_dir = out / 'predictions'
    prediction_dir.mkdir()
    result, skips, train_rows, test_rows, ranking_rows, backend_records = [], [], [], [], [], []
    for element in candidates:
        if f'ratio_{element}' not in database or element not in rankings:
            skips.append(dict(element=element, reason='missing fractions or ranking'))
            continue
        if group3(element):
            skips.append(dict(element=element, reason='group 3 excluded'))
            continue
        support = rankings[element]
        try:
            removed = exclusions(element, support, available)
            train_mask, test_mask = population_masks(database, element, support[:6], args.test_mode == 'filtered')
            test_ids = database.index[test_mask]
            if len(test_ids) <= args.min_test:
                raise ValueError(f'only {len(test_ids)} test compounds with f_X > {THRESHOLD}')
            # Check every pool before fitting any condition, keeping a complete cohort.
            sets = {}
            for repeat in range(args.repeats):
                for condition, excluded in removed.items():
                    ids = sample_pool(database, train_mask, excluded, args.train_size, args.seed + repeat)
                    validate_ids(database, ids, element, excluded, args.train_size)
                    sets[repeat, condition] = ids
        except ValueError as error:
            skips.append(dict(element=element, reason=str(error)))
            print(f'Skip {element}: {error}', flush=True)
            continue
        print(f'Held-out element {element}: {len(test_ids)} test compounds', flush=True)
        ranking_rows.append(dict(element=element, support_elements=json.dumps(support),
                                 support_A_XY=json.dumps(scores[element]), exclusions=json.dumps(removed)))
        test_hash = hashlib.sha256(json.dumps(test_ids.tolist()).encode()).hexdigest()
        test_rows.append(dict(element=element, n_ids=len(test_ids), ids=json.dumps(test_ids.tolist())))
        x_test = features.loc[test_ids].to_numpy(np.float32)
        y_test = database.loc[test_ids, args.task].to_numpy(np.float32)
        ratio_x = database.loc[test_ids, f'ratio_{element}'].to_numpy()
        # One gzip stream per element: only the current condition's predictions are held in memory.
        with gzip.open(prediction_dir / f'{element}.csv.gz', 'xt', encoding='utf-8', newline='') as prediction_file:
            first_chunk = True
            for repeat in range(args.repeats):
                baseline_mae = None
                for condition, excluded in removed.items():
                    ids = sets[repeat, condition]
                    model, backend = build_model(args.method, args.device, args.seed)
                    model.fit(features.loc[ids].to_numpy(np.float32), database.loc[ids, args.task].to_numpy(np.float32))
                    pred = model.predict(x_test)
                    if hasattr(pred, 'get'):
                        pred = pred.get()
                    if hasattr(pred, 'to_numpy'):
                        pred = pred.to_numpy()
                    pred = np.asarray(pred).reshape(-1)
                    if not np.isfinite(pred).all():
                        raise ValueError('Nonfinite model predictions')
                    score = metrics(y_test, pred)
                    if condition == 'Random':
                        baseline_mae = score['MAE']
                    row = dict(element=element, repeat=repeat, similarity=condition, count=len(test_ids),
                               n_train=len(ids), backend=backend, seed=args.seed + repeat,
                               test_ids_sha256=test_hash, delta_MAE=score['MAE'] - baseline_mae,
                               subset='threshold', lower=THRESHOLD,
                               support_top6=float(np.mean(scores[element][:6])), **score)
                    result.append(row)
                    backend_records.append(dict(element=element, repeat=repeat, condition=condition, backend=backend))
                    train_rows.append(dict(element=element, repeat=repeat, condition=condition,
                                           n_ids=len(ids), ids=json.dumps(ids.tolist())))
                    pd.DataFrame(dict(element=element, material_id=test_ids, repeat=repeat, condition=condition,
                                      true=y_test, prediction=pred, ratio_X=ratio_x)
                                 ).to_csv(prediction_file, index=False, header=first_chunk)
                    prediction_file.flush()
                    first_chunk = False
                    pd.DataFrame(result).to_csv(out / 'metrics.csv', index=False)
                    print(f'{element} repeat={repeat} {condition}: MAE={score["MAE"]:.8g}; '
                          f'ΔMAE={row["delta_MAE"]:.8g} ({backend})', flush=True)
    pd.DataFrame(skips, columns=['element', 'reason']).to_csv(out / 'skips.csv', index=False)
    pd.DataFrame(train_rows).to_csv(out / 'train_ids.csv.gz', index=False)
    pd.DataFrame(test_rows).to_csv(out / 'test_ids.csv.gz', index=False)
    pd.DataFrame(ranking_rows).to_csv(out / 'source_rankings.csv', index=False)
    write_json(out / 'backends.json', backend_records)
    if not result:
        raise RuntimeError('No eligible elements completed; inspect skips.csv')
    from summarize_ablation import summarize
    summary, per_element, correlations = summarize(pd.DataFrame(result), min_count=args.min_test)
    summary.to_csv(out / 'summary.csv', index=False)
    per_element.to_csv(out / 'support_error.csv', index=False)
    correlations.to_csv(out / 'correlations.csv', index=False)


if __name__ == '__main__':
    main()
