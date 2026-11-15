#!/usr/bin/env python
"""Rebuild the element vectors used by the embedding and elemental_properties measures.

Each element is represented by its single-element compound with the lowest formation energy.
  embedding:            OQMD layer-0 features of that compound, unscaled.
  elemental_properties: the 22 'MagpieData mean' Matminer columns of that compound in JARVIS,
                        standardized over all JARVIS compounds (not over the elements).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_io import atomic_order, load_database, load_features, standardize_global

SOURCES = {'embedding': ('OQMD', 'composition'), 'elemental_properties': ('Jarvis', 'matminer')}


def element_vectors(database, features):
    """Feature rows of each element's lowest-formation-energy single-element compound."""
    ratios = database.filter(regex='^ratio_')
    unary = database.loc[ratios.gt(0).sum(axis=1).eq(1) & database.e_form.notna()].copy()
    unary['element'] = ratios.loc[unary.index].idxmax(axis=1).str[6:]
    selected = unary.loc[unary.groupby('element', sort=False).e_form.idxmin()]
    # If the selected compound has no features, the element is omitted rather than substituted.
    selected = selected.loc[selected.index.isin(features.index)]
    vectors = features.loc[selected.index].copy()
    vectors.index = pd.Index(selected.element, name='element')
    return vectors.loc[atomic_order(vectors.index)]


def build(root, measure):
    db, representation = SOURCES[measure]
    database = load_database(root, db)
    features, _ = load_features(root, db, representation)
    if measure == 'elemental_properties':
        features, _ = standardize_global(features)
        columns = [c for c in features if c.startswith('MagpieData mean ')]
        if len(columns) != 22:
            raise ValueError(f'Expected 22 elemental properties, got {len(columns)}')
        features = features[columns]
    return element_vectors(database, features)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', type=Path, required=True, help='folder downloaded from the data repository')
    p.add_argument('--out-dir', type=Path, required=True, help='new or empty output folder')
    p.add_argument('--measures', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    args = p.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {args.out_dir}')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for measure in args.measures:
        vectors = build(args.data_dir, measure)
        path = args.out_dir / f'{measure}_vectors.csv'
        vectors.to_csv(path)
        message = f'Saved {path}: {vectors.shape[0]} elements x {vectors.shape[1]} values'
        published = args.data_dir / 'element_features' / path.name
        if published.exists():
            old = pd.read_csv(published, index_col=0, float_precision='round_trip')
            same = old.index.equals(vectors.index) and old.columns.equals(vectors.columns)
            difference = float(np.nanmax(np.abs(old.to_numpy() - vectors.to_numpy()))) if same else None
            message += (f'; max difference from published file: {difference:.3g}' if same
                        else '; elements or columns differ from the published file')
        print(message, flush=True)


if __name__ == '__main__':
    main()
