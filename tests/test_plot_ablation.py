"""Plot new runs and reference results, and enforce the strict count boundary."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ablation_core import parse_args
from plot_ablation import discover, load_result, main, prepare_series, series_labels
from summarize_ablation import summarize


def write_run(root, *, reference=False, task='e_form', count=101):
    root.mkdir(parents=True)
    frame = pd.DataFrame([
        dict(element='Ag', repeat=0, similarity=name, count=count, MAE=mae, subset='threshold', lower=.4)
        for name, mae in [('Random', .1), ('Top 2 excluded', .3), ('10-15 excluded' if reference else 'Ranks 11-16 excluded', .15)]])
    # These rows must not enter the plotted f_X>0.4 summary.
    other = frame.copy()
    other['lower'] = .1
    other['MAE'] = 10.
    frame = pd.concat([frame, other], ignore_index=True)
    config = dict(db='OQMD', task=task, method='SVR', representation='pcg', ranking='pcg',
                  test_mode='filtered', gpu=0)
    if reference:
        path = root / f'ablation_OQMD_{task}_all_eSEN_layer4_SVR_base10000_testtop6_abc.csv'
        config.pop('representation')
        config.pop('ranking')
        config.pop('test_mode')
        config.update(feature_source='esen', support_csv='results/axy/support_elements_OQMD_eSEN_layer4.csv')
        config_path = path.with_name(path.stem + '_config.json')
    else:
        path = root / 'metrics.csv'
        config_path = root / 'config.json'
        frame['backend'] = 'sklearn.SVR'
    frame.to_csv(path, index=False)
    config_path.write_text(json.dumps(config))
    return path


def test_reference_and_new_results_agree(tmp_path):
    old = write_run(tmp_path / 'reference', reference=True)
    new = write_run(tmp_path / 'new')
    a, old_metadata = prepare_series(old)
    b, new_metadata = prepare_series(new)
    np.testing.assert_allclose(a.mean_delta_MAE_meV, b.mean_delta_MAE_meV)
    assert a.set_index('similarity').loc['Top 2 excluded', 'mean_delta_MAE_meV'] == pytest.approx(200)
    assert a.set_index('similarity').loc['Ranks 11-16 excluded', 'mean_delta_MAE_meV'] == pytest.approx(50)
    assert old_metadata['ranking'] == new_metadata['ranking'] == 'pcg'
    assert old_metadata['test_mode'] == 'filtered'
    assert a.n_elements.eq(1).all() and a.n_repeats.eq(1).all()
    assert len(discover([tmp_path, new])) == 2


def test_strict_count_default_and_generation_default(tmp_path):
    path = write_run(tmp_path / 'run', count=100)
    assert prepare_series(path)[0].empty
    frame = pd.read_csv(path)
    frame['count'] = 101
    frame.to_csv(path, index=False)
    assert not prepare_series(path)[0].empty
    assert parse_args(['--data-dir', str(tmp_path), '--out-dir', str(tmp_path / 'out')]).min_test == 100
    assert summarize(frame)[0].n_observations.eq(1).all()
    frame['count'] = 100
    assert summarize(frame)[0].empty


def test_missing_baseline_and_duplicate_inputs_fail(tmp_path):
    path = write_run(tmp_path / 'run')
    frame = pd.read_csv(path)
    frame.loc[~frame.similarity.eq('Random')].to_csv(path, index=False)
    with pytest.raises(ValueError, match='baseline'):
        prepare_series(path)
    frame = frame.loc[frame.lower.eq(.4)]
    pd.concat([frame, frame.iloc[[0]]]).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Duplicate'):
        prepare_series(path)


def test_figure_outputs_and_cross_property_legend(tmp_path):
    energy = write_run(tmp_path / 'energy')
    gap = write_run(tmp_path / 'gap', task='bandgap')
    info = [prepare_series(path)[1] for path in (energy, gap)]
    labels, common = series_labels(info)
    assert labels == ['SVR', 'SVR']
    assert any('held out: Ag' == value for value in common)
    out = tmp_path / 'figures'
    main(['--inputs', str(energy.parent), str(gap), '--out-dir', str(out)])
    for name in ['ablation.png', 'ablation.pdf', 'plot_summary.csv', 'plot_config.json']:
        assert (out / name).stat().st_size > 0
    summary = pd.read_csv(out / 'plot_summary.csv')
    assert set(summary.task) == {'e_form', 'bandgap'}
    assert summary.series_id.nunique() == 1
    assert len(summary) == 6
    assert json.loads((out / 'plot_config.json').read_text())['settings']['min_count'] == 100
    with pytest.raises(FileExistsError):
        main(['--inputs', str(energy), '--out-dir', str(out)])


def test_filters_and_element_selection(tmp_path):
    path = write_run(tmp_path / 'run')
    assert prepare_series(path, elements=['O'])[0].empty
    with pytest.raises(ValueError, match='No eligible'):
        main(['--inputs', str(path), '--methods', 'RF', '--out-dir', str(tmp_path / 'figures')])
