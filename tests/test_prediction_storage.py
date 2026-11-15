"""Prediction rows are streamed into one gzip file per held-out element."""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import ablation_core
import regressors


def setup_small_run(tmp_path, monkeypatch, fail_on_fit=None):
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg',
                'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ag', 'Zn']
    database = pd.DataFrame(0., index=[str(i) for i in range(12)],
                            columns=[f'ratio_{e}' for e in elements])
    database.loc[[str(i) for i in range(8)], 'ratio_K'] = 1.
    database.loc[['8', '9'], 'ratio_Ag'] = 1.
    database.loc[['10', '11'], 'ratio_Zn'] = 1.
    database['e_form'] = np.arange(12, dtype=float) / 10
    features = pd.DataFrame(np.arange(12 * 128).reshape(12, 128), index=database.index)
    rankings = {target: [e for e in elements if e != target] for target in ('Ag', 'Zn')}
    scores = {target: np.linspace(.9, .1, len(source)).tolist() for target, source in rankings.items()}
    support = tmp_path / 'support.csv'
    support.write_text('fixture ranking; loader stubbed for this output-format test\n')
    monkeypatch.setattr(ablation_core, 'load_database', lambda *args: database.copy())
    monkeypatch.setattr(ablation_core, 'load_features', lambda *args: (features.copy(), {}))
    monkeypatch.setattr(ablation_core, 'read_rankings', lambda *args: (rankings, scores))
    fit_count = 0

    class Model:
        def fit(self, x, y):
            nonlocal fit_count
            fit_count += 1
            if fit_count == fail_on_fit:
                raise RuntimeError('simulated fit failure')
            return self

        def predict(self, x):
            return np.arange(len(x), dtype=float) + .25

    monkeypatch.setattr(regressors, 'build_model', lambda *args: (Model(), 'test-model'))
    out = tmp_path / 'run'
    argv = ['--data-dir', str(tmp_path), '--out-dir', str(out), '--support-csv', str(support),
            '--elements', 'Ag', 'Zn', '--repeats', '2', '--train-size', '3', '--min-test', '1']
    return out, argv, database


def test_all_repetitions_and_conditions_share_one_file_per_element(tmp_path, monkeypatch):
    out, argv, database = setup_small_run(tmp_path, monkeypatch)
    ablation_core.main(argv)
    assert sorted(p.name for p in (out / 'predictions').iterdir()) == ['Ag.csv.gz', 'Zn.csv.gz']
    assert not list(out.glob('predictions_*.csv.gz'))
    metrics = pd.read_csv(out / 'metrics.csv')
    assert len(metrics) == 2 * 2 * len(ablation_core.CONDITIONS)
    for element in ('Ag', 'Zn'):
        path = out / 'predictions' / f'{element}.csv.gz'
        frame = pd.read_csv(path, dtype={'material_id': str})
        assert frame.columns.tolist() == ['element', 'material_id', 'repeat', 'condition', 'true', 'prediction', 'ratio_X']
        assert len(frame) == 2 * 2 * len(ablation_core.CONDITIONS)
        assert frame.element.eq(element).all()
        assert frame.groupby(['repeat', 'condition']).size().eq(2).all()
        for repeat in range(2):
            assert frame.loc[frame['repeat'].eq(repeat), 'condition'].drop_duplicates().tolist() == list(ablation_core.CONDITIONS)
        assert not frame.duplicated(['element', 'material_id', 'repeat', 'condition']).any()
        for row in frame.itertuples():
            assert row.true == pytest.approx(np.float32(database.loc[row.material_id, 'e_form']))
            assert row.ratio_X == 1.
        for (repeat, condition), group in frame.groupby(['repeat', 'condition']):
            expected = np.mean(np.abs(group.true - group.prediction))
            actual = metrics.loc[metrics.element.eq(element) & metrics['repeat'].eq(repeat)
                                 & metrics.similarity.eq(condition), 'MAE'].item()
            assert actual == pytest.approx(expected)
        with gzip.open(path, 'rt') as handle:
            assert sum(line.startswith('element,material_id,repeat,condition,') for line in handle) == 1
    config = json.loads((out / 'config.json').read_text())
    assert config['prediction_storage']['path'] == 'predictions/{element}.csv.gz'


def test_completed_chunks_remain_readable_after_model_failure(tmp_path, monkeypatch):
    out, argv, _ = setup_small_run(tmp_path, monkeypatch, fail_on_fit=2)
    with pytest.raises(RuntimeError, match='simulated fit failure'):
        ablation_core.main(argv)
    frame = pd.read_csv(out / 'predictions/Ag.csv.gz')
    assert len(frame) == 2
    assert frame.condition.eq('Random').all()
    assert frame['repeat'].eq(0).all()
    assert not (out / 'predictions/Zn.csv.gz').exists()
