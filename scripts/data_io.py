"""Readers for the files in the data repository."""
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Element

DATABASES = ('Jarvis', 'MP', 'OQMD')
ID_COLUMNS = dict(Jarvis='jid', MP='material_id', OQMD='entry_id')
MEASURES = ('pcg', 'composition', 'embedding', 'elemental_properties')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values, indent=2, default=str, allow_nan=False) + '\n')


def canonical_ids(ids, db):
    result = pd.Index(ids).astype(str)
    if db == 'OQMD':
        result = result.str.replace(r'^tensor\(([0-9]+)\)$', r'\1', regex=True)
        if not result.str.fullmatch(r'[0-9]+').all():
            raise ValueError('Invalid OQMD entry_id')
        result = result.astype(np.int64).astype(str)
    return result.rename(ID_COLUMNS[db])


def load_database(root, db):
    path = Path(root) / 'databases' / f'{db}.csv'
    key = ID_COLUMNS[db]
    frame = pd.read_csv(path, usecols=lambda c: c == key or c.startswith('ratio_')
                        or c in ('e_form', 'bandgap', 'formula'), dtype={key: str})
    if frame[key].isna().any():
        raise ValueError('Missing compound IDs')
    frame.index = canonical_ids(frame.pop(key), db)
    if not frame.index.is_unique:
        raise ValueError(f'Duplicate database IDs in {path}')
    ratios = frame.filter(regex='^ratio_').to_numpy(float)
    if not np.isfinite(ratios).all() or ((ratios < 0) | (ratios > 1)).any():
        raise ValueError('Element fractions must be finite and between zero and one')
    if not np.allclose(ratios.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Element fractions do not sum to one')
    return frame


def feature_files(root, db, representation):
    root = Path(root)
    if representation == 'matminer':
        return [root / 'features/matminer' / f'{db}.pkl']
    folder = {'pcg': 'pcg_layer4', 'composition': 'composition_layer0'}[representation]
    return [root / 'features' / folder / f'{db}_{part}.npz' for part in ('train', 'test')]


def load_features(root, db, representation):
    paths = feature_files(root, db, representation)
    if representation == 'matminer':
        # Pickle from the data repository (load only trusted files); the last 273 columns are the descriptors.
        original = pd.read_pickle(paths[0])
        features = original.iloc[:, -273:].astype(np.float64).copy()
        del original
    else:
        frames = []
        for path in paths:
            with np.load(path, allow_pickle=True) as raw:
                values = np.vstack(raw['features']).astype(np.float64)
                frames.append(pd.DataFrame(values, index=raw['material_ids'],
                                           columns=[f'feat_{i}' for i in range(values.shape[1])]))
        features = pd.concat(frames)
    features.index = canonical_ids(features.index, db)
    n_duplicates = int(features.index.duplicated().sum())
    if n_duplicates:
        duplicates = features.index[features.index.duplicated()].unique()
        for key in duplicates:
            rows = features.loc[[key]].to_numpy()
            if not np.allclose(rows, rows[0], equal_nan=True):
                raise ValueError(f'Conflicting duplicate feature ID: {key}')
        features = features.loc[~features.index.duplicated(keep='first')]
    expected = 273 if representation == 'matminer' else 128
    if features.shape[1] != expected:
        raise ValueError(f'{representation}: expected {expected} features, got {features.shape[1]}')
    if np.isinf(features.to_numpy()).any():
        raise ValueError('Infinite feature values')
    report = dict(rows=len(features), columns=features.shape[1], duplicate_rows=n_duplicates,
                  missing_values=int(features.isna().sum().sum()),
                  all_missing_rows=int(features.isna().all(axis=1).sum()))
    print(f'{db}/{representation}: {report}', flush=True)
    return features, report


def standardize_global(features):
    """Standardize each feature over all compounds (pandas std, ddof=1); used for label-free geometry."""
    mean = features.mean()
    scale = features.std(ddof=1)
    scale = scale.mask(scale < 1e-12, 1.0)
    return (features - mean) / scale, pd.DataFrame({'mean': mean, 'scale': scale})


def align(database, features, drop_all_nan=False):
    ids = database.index[database.index.isin(features.index)]
    data, feats = database.loc[ids], features.loc[ids]
    all_nan = feats.isna().all(axis=1)
    report = dict(database_rows=len(database), feature_rows=len(features), aligned_rows=len(ids),
                  database_without_features=int((~database.index.isin(features.index)).sum()),
                  features_without_database=int((~features.index.isin(database.index)).sum()),
                  all_missing_rows=int(all_nan.sum()), filled_nan_values=int(feats.isna().sum().sum()))
    if drop_all_nan:
        data, feats = data.loc[~all_nan], feats.loc[~all_nan]
    return data, feats.fillna(0), report


def ranking_path(root, measure='pcg', probe_db='OQMD'):
    folder = Path(root) / 'axy' / measure
    if measure in ('embedding', 'elemental_properties'):
        return folder / 'support_elements_df.csv'
    layer = 4 if measure == 'pcg' else 0
    return folder / f'support_elements_{probe_db}_eSEN_layer{layer}.csv'


def read_rankings(path):
    frame = pd.read_csv(path, dtype={'leaveout': str})
    if not frame.leaveout.is_unique:
        raise ValueError('Duplicate ranking targets')
    rankings, scores = {}, {}
    for row in frame.itertuples():
        elements = ast.literal_eval(row.support_elements)
        values = ast.literal_eval(row.support_A_XY)
        if (len(elements) != len(values) or len(set(elements)) != len(elements)
                or row.leaveout in elements or not all(isinstance(e, str) for e in elements)
                or not np.isfinite(values).all() or np.any(np.diff(values) > 1e-12)):
            raise ValueError(f'Invalid source ranking for {row.leaveout}')
        rankings[row.leaveout], scores[row.leaveout] = elements, values
    return rankings, scores


def atomic_order(elements):
    return sorted(elements, key=lambda e: Element(e).Z)


def group3(element):
    return Element(element).group == 3


def same_group(element, available):
    return atomic_order([e for e in available if e != element and Element(e).group == Element(element).group])
