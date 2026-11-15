"""Compute A_XY relationships and source rankings from features or element vectors."""
import argparse
from pathlib import Path

import pandas as pd

from data_io import (DATABASES, MEASURES, align, feature_files, load_database, load_features,
                     ranking_path, read_rankings, sha256, standardize_global, write_json)
from trajectories import fit_trajectory, pairwise


def compare_tables(pairs, ranks, published_pairs, published_ranks):
    """Compare new values and top-six sources with the published files."""
    old = pd.read_csv(published_pairs)
    merged = pairs.merge(old[['element_X', 'element_Y', 'A_XY']], on=['element_X', 'element_Y'],
                         suffixes=('_new', '_published'))
    merged['difference'] = merged.A_XY_new - merged.A_XY_published
    old_rank, _ = read_rankings(published_ranks)
    rows = []
    for row in ranks.itertuples():
        if row.leaveout not in old_rank:
            continue
        new, old_el = row.support_elements, old_rank[row.leaveout]
        rows.append(dict(element=row.leaveout, top6_same_set=set(new[:6]) == set(old_el[:6]),
                         top6_same_order=new[:6] == old_el[:6],
                         n_top6_changed=len(set(new[:6]).symmetric_difference(old_el[:6])),
                         new_top6=' '.join(new[:6]), published_top6=' '.join(old_el[:6])))
    return merged, pd.DataFrame(rows)


def compute(root, out, db, measure, min_bin_count=20):
    root, out = Path(root), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    config = dict(db=db, measure=measure, min_bin_count=min_bin_count,
                  scaling='features standardized over all compounds of the dataset (ddof=1)',
                  self_pairs='empty, excluded from ranking', tie_break='atomic number')
    if measure in ('embedding', 'elemental_properties'):
        path = root / 'element_features' / f'{measure}_vectors.csv'
        frame = pd.read_csv(path, index_col=0)
        vectors = {e: row.to_numpy(float) for e, row in frame.iterrows()}
        config.update(db=None, vector_file=str(path.relative_to(root)), vector_sha256=sha256(path),
                      scaling='vectors used as supplied')
    else:
        database = load_database(root, db)
        features, report = load_features(root, db, measure)
        features, scaling = standardize_global(features)
        scaling.to_csv(out / 'scaling.csv', index_label='feature')
        database, features, alignment = align(database, features, drop_all_nan=True)
        config.update(features=report, alignment=alignment,
                      inputs=[dict(file=str(p.relative_to(root)), sha256=sha256(p)) for p in feature_files(root, db, measure)])
        values = features.to_numpy(float)
        vectors, bins, trajectories, centroids = {}, [], [], []
        # A_XY uses no property labels: fit every element on the complete probe collection.
        for column in database.filter(regex='^ratio_'):
            element = column[6:]
            ratios = database[column].to_numpy(float)
            present = ratios > 0
            if not present.any() or ratios[present].var() < 1e-12:
                continue
            slope, detail, centers = fit_trajectory(values[present], ratios[present], min_bin_count)
            bins.extend(dict(element=element, **row) for row in detail)
            if slope is None:
                continue
            vectors[element] = slope
            trajectories.append(dict(element=element, n_bins_used=sum(b['used'] for b in detail),
                                     direction=str(slope.tolist())))
            for row, vector in zip([b for b in detail if b['used']], centers):
                centroids.append(dict(element=element, center=row['center'], count=row['count'],
                                      centroid=str(vector.tolist())))
        pd.DataFrame(trajectories).to_csv(out / 'directions.csv', index=False)
        pd.DataFrame(bins).to_csv(out / 'bins.csv', index=False)
        pd.DataFrame(centroids).to_csv(out / 'centroids.csv', index=False)
    pairs, ranks = pairwise(vectors)
    pairs.to_csv(out / 'Axy.csv', index=False)
    ranks.to_csv(out / 'support_elements.csv', index=False)
    published_ranks = ranking_path(root, measure, db)
    published_pairs = (published_ranks.with_name('axy_pairs_df.csv') if measure in ('embedding', 'elemental_properties')
                       else published_ranks.with_name(published_ranks.name.replace('support_elements_', 'Axy_')))
    if published_pairs.exists() and published_ranks.exists():
        differences, memberships = compare_tables(pairs, ranks, published_pairs, published_ranks)
        differences.to_csv(out / 'comparison_values.csv', index=False)
        memberships.to_csv(out / 'comparison_top6.csv', index=False)
        finite = differences.difference.dropna()
        config['published_comparison'] = dict(n_shared_pairs=len(finite),
            max_abs_difference=float(finite.abs().max()) if len(finite) else None,
            mean_abs_difference=float(finite.abs().mean()) if len(finite) else None,
            top6_sets_changed=int((~memberships.top6_same_set).sum()))
    write_json(out / 'config.json', config)
    label = measure if config['db'] is None else f'{measure}/{db}'
    print(f'Saved {label} relationships for {len(vectors)} elements to {out}', flush=True)
    if 'published_comparison' in config:
        print(f"Compared with published files: {config['published_comparison']}", flush=True)
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True, help='folder downloaded from the data repository')
    p.add_argument('--out-dir', type=Path, required=True, help='new or empty output folder')
    p.add_argument('--db', choices=DATABASES, default='OQMD', help='probe dataset (pcg and composition only)')
    p.add_argument('--measure', choices=MEASURES, default='pcg')
    p.add_argument('--min-bin-count', type=int, default=20, help='minimum compounds per fraction interval')
    args = p.parse_args()
    if args.min_bin_count < 1:
        p.error('min-bin-count must be positive')
    compute(args.data_dir, args.out_dir, args.db, args.measure, args.min_bin_count)


if __name__ == '__main__':
    main()
