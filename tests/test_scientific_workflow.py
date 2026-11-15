"""Small tests of the scientific invariants, independent of release data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ablation_core import exclusions, population_masks, sample_pool, validate_ids
from data_io import align, canonical_ids, standardize_global
from element_vectors import element_vectors
from regressors import build_model
from summarize_ablation import summarize, trimmed_mean
from trajectories import EDGES, fit_trajectory, pairwise


def test_bin_boundaries_and_actual_centers():
    fractions = np.array([0, EDGES[1], EDGES[2], EDGES[3], EDGES[4], EDGES[5]])
    values = np.column_stack([fractions, 2 * fractions])
    slope, bins, _ = fit_trajectory(values, fractions, min_bin_count=1)
    assert [b['count'] for b in bins] == [1, 1, 1, 1, 1]
    np.testing.assert_allclose(slope, [1, 2])


def test_missing_bins_use_actual_centers():
    fractions = np.array([0.1, 0.5, 0.7, 0.9])
    values = np.column_stack([fractions, fractions ** 2])
    actual, _, _ = fit_trajectory(values, fractions, 1)
    expected = np.linalg.lstsq(np.column_stack([np.ones(4), fractions]), values, rcond=None)[0][1]
    np.testing.assert_allclose(actual, expected)
    assert fit_trajectory(values[:2], fractions[:2], 1)[0] is None


def test_cosine_invalid_self_and_atomic_ties():
    pairs, ranks = pairwise({'He': np.array([1., 0]), 'H': np.array([1., 0]),
                             'Li': np.array([1., 0]), 'Be': np.array([0., 0])})
    he = ranks.set_index('leaveout').loc['He']
    assert he.support_elements == ['H', 'Li']
    assert he.support_A_XY == [1., 1.]
    assert pairs.loc[pairs.element_X.eq(pairs.element_Y), 'A_XY'].isna().all()
    assert ranks.set_index('leaveout').loc['Be'].support_elements == []


def test_scaling_alignment_and_canonical_ids():
    features = pd.DataFrame([[0., 1], [2., 1], [np.nan, 1]], index=['a', 'b', 'c'])
    scaled, scale = standardize_global(features)
    assert scale.loc[1, 'scale'] == 1
    assert scaled.loc['a', 0] == pytest.approx(-1 / np.sqrt(2))
    database = pd.DataFrame({'e_form': [0., 1.]}, index=['b', 'a'])
    meta, aligned, report = align(database, features)
    assert aligned.index.tolist() == meta.index.tolist() == ['b', 'a']
    assert report['features_without_database'] == 1
    assert canonical_ids(['tensor(123)', 456], 'OQMD').tolist() == ['123', '456']


def test_exclusion_filter_sampling_and_id_validation():
    database = pd.DataFrame({'ratio_Ag': [0, 0, 0, .5, .5], 'ratio_H': [1, 0, 0, .5, 0],
                             'ratio_Li': [0, 1, 0, 0, .5], 'ratio_Na': [0, 0, 1, 0, 0]},
                            index=['a', 'b', 'c', 'd', 'e'])
    train, test = population_masks(database, 'Ag', ['H'], True)
    assert database.index[test].tolist() == ['e']
    assert database.index[train].tolist() == ['a', 'b', 'c']
    ids = sample_pool(database, train, ['H'], 2, 42)
    validate_ids(database, ids, 'Ag', ['H'], 2)
    np.testing.assert_array_equal(ids, sample_pool(database, train, ['H'], 2, 42))
    with pytest.raises(ValueError, match='requires exactly'):
        sample_pool(database, train, ['H'], 3, 42)
    with pytest.raises(ValueError, match='leaked'):
        validate_ids(database, pd.Index(['a', 'b']), 'Ag', ['H'], 2)


def test_training_only_svr_scaling():
    model, backend = build_model('SVR')
    model.fit(np.array([[0., 1], [2., 1], [4., 1]], np.float32), [0., 1., 2.])
    np.testing.assert_allclose(model[0].mean_, [2., 1.])
    model.predict(np.array([[10000., 10000.]]))
    np.testing.assert_allclose(model[0].mean_, [2., 1.])
    assert backend == 'sklearn.SVR'
    assert model[1].C == 1 and model[1].epsilon == .1 and model[1].gamma == 'scale'


def metric_fixture():
    rows = []
    for element, support, baseline in [('H', .1, 3.), ('Li', .5, 2.), ('Na', .9, 1.)]:
        for repeat in range(2):
            for condition, value in [('Random', baseline), ('Top 6 excluded', baseline + .2)]:
                rows.append(dict(element=element, repeat=repeat, similarity=condition,
                                 count=300, MAE=value, support_top6=support))
    return pd.DataFrame(rows)


def test_paired_deltas_trim_counts_and_correlations():
    assert trimmed_mean([0, 1, 2, 3, 4, 5, 6, 7, 8, 100], .1) == (4.5, 8)
    summary, per, corr = summarize(metric_fixture())
    assert summary.set_index('similarity').loc['Top 6 excluded', 'mean_delta_MAE'] == pytest.approx(.2)
    assert corr.iloc[0].spearman_rho == pytest.approx(-1)
    assert corr.iloc[0].n_elements == 3
    assert summarize(metric_fixture(), min_count=300)[0].empty
    group3_fixture = metric_fixture().replace({'element': {'Na': 'La'}})
    assert summarize(group3_fixture)[0].n_elements.eq(2).all()
    assert summarize(group3_fixture, include_group3=True)[0].n_elements.eq(3).all()
    broken = metric_fixture().drop(index=0)
    with pytest.raises(ValueError, match='baseline'):
        summarize(broken)
    broken = metric_fixture()
    broken.loc[1, 'count'] = 301
    with pytest.raises(ValueError, match='counts'):
        summarize(broken)
    broken = metric_fixture()
    broken['test_ids_sha256'] = 'same_ids'
    broken.loc[1, 'test_ids_sha256'] = 'different_ids'
    with pytest.raises(ValueError, match='identifiers'):
        summarize(broken)


def test_element_vectors_use_lowest_energy_unary_compound():
    database = pd.DataFrame({
        'ratio_H': [1, 1, 0, 0, 0, .5],
        'ratio_Li': [0, 0, 1, 0, 0, .5],
        'ratio_Na': [0, 0, 0, 1, 1, 0],
        'e_form': [.2, 0., 0., -.1, np.nan, -1.]},
        index=['h1', 'h2', 'li1', 'na1', 'na2', 'mix'])
    features = pd.DataFrame(np.arange(10.).reshape(5, 2), index=['h1', 'h2', 'na1', 'na2', 'mix'])
    vectors = element_vectors(database, features)
    # Na sorts after H by atomic number; Li is omitted because its selected compound has no features.
    assert vectors.index.tolist() == ['H', 'Na']
    np.testing.assert_array_equal(vectors.loc['H'], features.loc['h2'])
    np.testing.assert_array_equal(vectors.loc['Na'], features.loc['na1'])
