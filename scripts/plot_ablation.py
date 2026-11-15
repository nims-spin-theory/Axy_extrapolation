#!/usr/bin/env python
"""Plot source-exclusion results (new runs or reference results) without training models.

Examples:
  python plot_ablation.py --inputs /path/to/new-run --out-dir /path/to/plots
  python plot_ablation.py --inputs /path/to/data/reference_results/ablation_filtered \
      --db OQMD --methods RF XGB SVR --rankings pcg --out-dir /path/to/plots
"""
import argparse
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ablation_core import CONDITIONS
from data_io import DATABASES, MEASURES, group3, sha256, write_json
from summarize_ablation import summarize

REFERENCE_NAME = re.compile(r'^ablation_(Jarvis|MP|OQMD)_(e_form|bandgap)_all_'
                          r'(eSEN_layer4|MatMiner)_(RF|XGB|SVR)_base')
ALIASES = {'Baseline': 'Random', '10-15 excluded': 'Ranks 11-16 excluded'}
DISPLAY = {'db': 'dataset', 'method': 'regressor', 'representation': 'features',
           'ranking': 'ranking', 'test_mode': 'tests', 'backend': 'backend',
           'n_elements': 'held-out elements', 'n_repeats': 'repetitions'}
PROPERTIES = {'e_form': ('Formation energy', 'meV/atom'), 'bandgap': ('Bandgap', 'meV')}


def discover(inputs):
    paths = set()
    for item in inputs:
        item = Path(item).resolve()
        if item.is_file():
            paths.add(item)
        elif item.is_dir():
            if (item / 'metrics.csv').is_file():
                paths.add(item / 'metrics.csv')
            else:
                paths.update(item.rglob('metrics.csv'))
                paths.update(p for p in item.rglob('ablation_*.csv') if REFERENCE_NAME.match(p.name))
        else:
            raise FileNotFoundError(item)
    if not paths:
        raise ValueError('No element-level result CSVs found; provide a run directory, reference-result directory, or result CSV')
    return sorted(paths)


def metadata_for(path):
    config_path = path.parent / 'config.json' if path.name == 'metrics.csv' else path.with_name(path.stem + '_config.json')
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    metadata = {}
    match = REFERENCE_NAME.match(path.name)
    if match:
        db, task, representation, method = match.groups()
        metadata.update(db=db, task=task, representation='pcg' if representation == 'eSEN_layer4' else 'matminer', method=method)
    for key in ('db', 'task', 'method', 'representation', 'ranking', 'test_mode'):
        if key in config:
            metadata[key] = config[key]
    if 'representation' not in metadata:
        metadata['representation'] = {'esen': 'pcg', 'matminer': 'matminer'}.get(config.get('feature_source'))
    support = str(config.get('support_csv', ''))
    if 'ranking' not in metadata:
        if 'magpie' in support or 'elemental_properties' in support:
            metadata['ranking'] = 'elemental_properties'
        elif 'embedding' in support:
            metadata['ranking'] = 'embedding'
        elif 'layer0' in support or 'composition' in support:
            metadata['ranking'] = 'composition'
        elif 'layer4' in support:
            metadata['ranking'] = 'pcg'
        elif path.parent.name in MEASURES:
            metadata['ranking'] = path.parent.name
    metadata.setdefault('test_mode', 'filtered' if 'testtop' in path.name else 'unfiltered')
    for key in ('db', 'task', 'method', 'representation', 'ranking'):
        if not metadata.get(key):
            raise ValueError(f'{path}: cannot infer {key}; keep the companion config.json or *_config.json with the CSV')
    if metadata['task'] not in PROPERTIES:
        raise ValueError(f'Unsupported property: {metadata["task"]}')
    metadata['backend'] = ('cuml.' if config.get('gpu') else 'sklearn.') + metadata['method']
    if metadata['method'] == 'XGB':
        metadata['backend'] = 'xgboost.' + ('gpu' if config.get('gpu') else 'cpu')
    metadata['source'] = str(path)
    metadata['support_csv'] = support
    metadata['support_sha256'] = config.get('support_sha256', config.get('support_csv_sha256', ''))
    return metadata


def load_result(path):
    metadata = metadata_for(path)
    frame = pd.read_csv(path, dtype={'element': str})
    required = {'element', 'repeat', 'similarity', 'MAE', 'count', 'subset', 'lower'}
    if not required.issubset(frame):
        raise ValueError(f'{path}: missing element-level result columns {sorted(required.difference(frame))}')
    frame['similarity'] = frame.similarity.replace(ALIASES)
    if 'backend' in frame:
        metadata['backend'] = '/'.join(sorted(frame.backend.dropna().astype(str).unique()))
    return frame, metadata


def prepare_series(path, *, elements=None, min_count=100, trim=0.1, include_group3=False, threshold=0.4):
    frame, metadata = load_result(path)
    if elements:
        frame = frame.loc[frame.element.isin(elements)]
    summary, _, _ = summarize(frame, min_count=min_count, trim=trim, include_group3=include_group3,
                               threshold=threshold)
    if summary.empty:
        return summary, metadata
    eligible = frame.loc[frame['subset'].eq('threshold') & np.isclose(frame.lower, threshold)
                         & frame['count'].gt(min_count)]
    if not include_group3:
        eligible = eligible.loc[~eligible.element.map(group3)]
    metadata['n_elements'] = str(eligible.element.nunique())
    metadata['n_repeats'] = str(eligible['repeat'].nunique())
    metadata['elements'] = sorted(eligible.element.unique())
    metadata['source_sha256'] = sha256(path)
    summary['n_repeats'] = summary.similarity.map(eligible.groupby('similarity')['repeat'].nunique())
    for key in ('db', 'task', 'method', 'representation', 'ranking', 'test_mode', 'backend', 'source'):
        summary[key] = metadata[key]
    return summary, metadata


def format_field(key, value):
    if key == 'representation':
        return f'{"PCG" if value == "pcg" else "MatMiner"} features'
    if key == 'ranking':
        value = {'pcg': 'PCG', 'composition': 'composition', 'embedding': 'embedding',
                 'elemental_properties': 'elemental-property'}[value]
        return f'{value} ranking'
    if key == 'test_mode':
        return f'{value} tests'
    if key in ('n_elements', 'n_repeats'):
        noun = DISPLAY[key][:-1] if value == '1' else DISPLAY[key]
        return f'{value} {noun}'
    if key == 'backend':
        if value.startswith('sklearn.'):
            return 'CPU (sklearn)'
        if value.startswith('cuml.'):
            return 'GPU (cuML)'
        if value.startswith('xgboost.'):
            return f'{value.split(".")[-1].upper()} (XGBoost)'
    return str(value)


def series_labels(metadata):
    keys = list(DISPLAY)
    varying = [key for key in keys if len({m[key] for m in metadata}) > 1]
    common = [format_field(key, metadata[0][key]) for key in keys if key not in varying]
    if 'backend' in varying and all(len({m['backend'] for m in metadata if m['method'] == method}) == 1
                                    for method in {m['method'] for m in metadata}):
        varying.remove('backend')
    labels = [' | '.join(format_field(key, m[key]) for key in varying) if varying else m['method']
              for m in metadata]
    for label in set(labels):
        indices = [i for i, value in enumerate(labels) if value == label]
        by_task = {}
        for index in indices:
            by_task.setdefault(metadata[index]['task'], []).append(index)
        if any(len(group) > 1 for group in by_task.values()):
            for group in by_task.values():
                for run, index in enumerate(group, 1):
                    labels[index] += f' | run {run}'
    if all(m['n_elements'] == '1' and m['elements'] == metadata[0]['elements'] for m in metadata):
        common.append('held out: ' + metadata[0]['elements'][0])
    return labels, common


def render(summaries, metadata, out, *, title=None, trim=0.1, threshold=0.4, min_count=100):
    labels, common = series_labels(metadata)
    legend_labels = list(dict.fromkeys(labels))
    tasks = [task for task in PROPERTIES if any(m['task'] == task for m in metadata)]
    conditions = [c for c in CONDITIONS if any(c in set(s.similarity) for s in summaries)]
    conditions.extend(sorted(set().union(*(set(s.similarity) for s in summaries)).difference(conditions)))
    legend_rows = math.ceil(len(legend_labels) / min(3, len(legend_labels)))
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(9, 7 * len(tasks)), 5.6 + legend_rows * .25), squeeze=False)
    handles = []
    offsets = np.linspace(-.28, .28, len(legend_labels)) if len(legend_labels) > 1 else [0.]
    palette = plt.get_cmap('tab10')
    markers = ('o', 's', '^', 'D', 'v', 'P', 'X')
    plotted = []
    for i, (summary, info, label) in enumerate(zip(summaries, metadata, labels)):
        style = legend_labels.index(label)
        color, marker = palette(style % 10), markers[(style // 10) % len(markers)]
        if label not in [h.get_label() for h in handles]:
            handles.append(Line2D([], [], color=color, marker=marker, linestyle='none', label=label))
        for ax, task in zip(axes[0], tasks):
            if info['task'] != task:
                continue
            selected = summary.set_index('similarity').reindex(conditions)
            ax.scatter(np.arange(len(conditions)) + offsets[style], selected.mean_delta_MAE_meV,
                       color=color, marker=marker, s=48, zorder=3)
        selected = summary.copy()
        selected['series'] = label
        selected['series_id'] = style + 1
        plotted.append(selected)
    tick_labels = {'Random': 'Baseline', 'Top 2 excluded': 'Top 2\nexcluded', 'Top 4 excluded': 'Top 4\nexcluded',
                   'Top 6 excluded': 'Top 6\nexcluded', 'Ranks 11-16 excluded': 'Ranks\n11–16\nexcluded',
                   'Same group excluded': 'Same group\nexcluded'}
    for ax, task in zip(axes[0], tasks):
        name, unit = PROPERTIES[task]
        ax.set_title(name)
        ax.set_ylabel(f'ΔMAE relative to baseline ({unit})')
        ax.set_xticks(np.arange(len(conditions)), [tick_labels.get(c, c) for c in conditions], fontsize=9)
        ax.set_xlabel('Source-exclusion condition')
        ax.axhline(0, color='.45', linewidth=.9, linestyle='--', zorder=1)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title or 'Source-exclusion ablation', fontsize=15, y=.985)
    subtitle = ' | '.join(common)
    subtitle += f'\nf_X > {threshold:g}; test count > {min_count}; {100 * trim:g}% trimmed from each tail'
    fig.text(.5, .935, '\n'.join(textwrap.fill(line, 125) for line in subtitle.splitlines()),
             ha='center', va='top', fontsize=9)
    bottom = (legend_rows * .27 + .25) / fig.get_figheight()
    fig.legend(handles=handles, loc='lower center', ncol=min(3, len(legend_labels)), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, bottom, 1, .85))
    fig.savefig(out / 'ablation.png', dpi=220, bbox_inches='tight')
    fig.savefig(out / 'ablation.pdf', bbox_inches='tight')
    plt.close(fig)
    result = pd.concat(plotted, ignore_index=True)
    result.to_csv(out / 'plot_summary.csv', index=False)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--inputs', nargs='+', type=Path, required=True, help='result CSVs, new run directories, or reference-result directories')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--db', nargs='+', choices=DATABASES)
    p.add_argument('--tasks', nargs='+', choices=list(PROPERTIES))
    p.add_argument('--methods', nargs='+', choices=['RF', 'XGB', 'SVR'])
    p.add_argument('--representations', nargs='+', choices=['pcg', 'matminer'])
    p.add_argument('--rankings', nargs='+', choices=MEASURES)
    p.add_argument('--test-mode', choices=['filtered', 'unfiltered'])
    p.add_argument('--elements', nargs='+')
    p.add_argument('--min-count', type=int, default=100, help='strict test-count cutoff')
    p.add_argument('--threshold', type=float, default=.4)
    p.add_argument('--trim', type=float, default=.1, help='fraction removed from EACH tail')
    p.add_argument('--include-group3', action='store_true')
    p.add_argument('--title')
    args = p.parse_args(argv)
    if args.min_count < 0 or not 0 <= args.trim < .5 or not 0 <= args.threshold < 1:
        p.error('require min-count>=0, 0<=trim<0.5, and 0<=threshold<1')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f'Plot output directory must be empty: {args.out_dir}')
    summaries, metadata, skipped = [], [], []
    filters = dict(db=args.db, task=args.tasks, method=args.methods, representation=args.representations,
                   ranking=args.rankings, test_mode=[args.test_mode] if args.test_mode else None)
    for path in discover(args.inputs):
        info = metadata_for(path)
        if any(values is not None and info[key] not in values for key, values in filters.items()):
            continue
        summary, info = prepare_series(path, elements=args.elements, min_count=args.min_count,
            trim=args.trim, include_group3=args.include_group3, threshold=args.threshold)
        if summary.empty:
            skipped.append(dict(source=str(path), reason='no rows after threshold, count, and element selection'))
            continue
        summaries.append(summary)
        metadata.append(info)
    if not summaries:
        raise ValueError('No eligible results match the requested filters; provide element-level CSVs and check the count/threshold settings')
    order = sorted(range(len(metadata)), key=lambda i: (
        list(PROPERTIES).index(metadata[i]['task']), DATABASES.index(metadata[i]['db']),
        metadata[i]['representation'], MEASURES.index(metadata[i]['ranking']),
        ['RF', 'XGB', 'SVR'].index(metadata[i]['method']), metadata[i]['source']))
    summaries, metadata = [summaries[i] for i in order], [metadata[i] for i in order]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = render(summaries, metadata, args.out_dir, title=args.title, trim=args.trim,
                    threshold=args.threshold, min_count=args.min_count)
    write_json(args.out_dir / 'plot_config.json', dict(settings=vars(args), series=metadata, skipped=skipped,
               aggregation='paired element/repeat ΔMAE; pooled trimmed mean; no cross-file pooling',
               error_bars='none; repetition/element counts are reported instead'))
    print(f'Saved ablation.png, ablation.pdf, plot_summary.csv, plot_config.json: {args.out_dir}')
    print(f'{len(metadata)} series; {len(result)} condition summaries. No models were trained.')


if __name__ == '__main__':
    main()
