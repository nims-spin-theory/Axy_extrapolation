# A<sub>XY</sub> chemical extrapolation

Code for *Materials representations organize transferable supervision for chemical extrapolation* by Enda Xiao and Terumasa Tadano.

In leave-one-element-out extrapolation, compounds containing a held-out element X are excluded from training and used for testing. A<sub>XY</sub>, computed from a materials representation without property labels, ranks the other elements Y by the potential of their compounds to provide transferable supervision for X. This repository lets you:

- compute A<sub>XY</sub> and the resulting source rankings,
- run fixed-size source-exclusion experiments that test these rankings,
- summarize and plot new results and the published reference results.

The data, including precomputed representations, rankings and 180 reference experiments, are on Hugging Face: [nims-spin-theory/Axy_extrapolation](https://huggingface.co/datasets/nims-spin-theory/Axy_extrapolation). 

## Installation

```bash
git clone https://github.com/nims-spin-theory/Axy_extrapolation.git
cd Axy_extrapolation
conda env create -f environment.yml
conda activate axy
```

The code was tested with Python 3.9 and the versions pinned in `requirements.txt`. Run all commands from the repository root.

The code runs on the CPU by default. To use an NVIDIA GPU (CUDA 12), also install the GPU backends and add `--device gpu` to the commands:

```bash
python -m pip install -r requirements-gpu.txt
```

The CPU (scikit-learn) and GPU (cuML) backends give slightly different MAE values. In our reruns, individual MAE values differed from the reference results by at most 0.4 meV/atom.

## Data

Download the files you need from the [data repository](https://huggingface.co/datasets/nims-spin-theory/Axy_extrapolation), keeping its folder structure. The quick start needs four files (1.6 GB):

```python
from huggingface_hub import snapshot_download  # pip install huggingface_hub

snapshot_download(
    repo_id="nims-spin-theory/Axy_extrapolation",
    repo_type="dataset",
    local_dir="Axy_extrapolation_data",
    allow_patterns=[
        "databases/OQMD.csv",
        "features/pcg_layer4/OQMD_*.npz",
        "axy/pcg/support_elements_OQMD_eSEN_layer4.csv",
    ],
)
```

Add `"reference_results/*"` (44 MB) to compare with the reference results. Then point the commands to the download folder:

```bash
export AXY_DATA_DIR=/absolute/path/to/Axy_extrapolation_data
```

## Quick start: one held-out element

Predict OQMD formation energies with PCG features and SVR, holding out Ca, for all six training conditions and five repetitions:

```bash
python scripts/run_ablation.py \
  --data-dir "$AXY_DATA_DIR" --out-dir output/ca \
  --db OQMD --task e_form --representation pcg --ranking pcg \
  --method SVR --elements Ca --repeats 5
```

The six conditions are:

| Condition | Training pool (10,000 compounds sampled from it) |
|---|---|
| `Random` | All compounds without Ca (baseline) |
| `Top 2 excluded`, `Top 4 excluded`, `Top 6 excluded` | Without compounds containing Ca's 2, 4 or 6 highest-ranked source elements (Sr, Yb, Ba, Eu, Lu, Y) |
| `Ranks 11-16 excluded` | Without compounds containing the source elements ranked 11–16 (control) |
| `Same group excluded` | Without compounds containing the other group-2 elements Be, Mg, Sr, Ba and Ra (control) |

Plot the result next to the reference result for Ca (requires `reference_results/`):

```bash
python scripts/plot_ablation.py --elements Ca --out-dir output/ca_figures \
  --inputs output/ca \
  "$AXY_DATA_DIR/reference_results/ablation_filtered/pcg/ablation_OQMD_e_form_all_eSEN_layer4_SVR_base10000_testtop6_81ecfc423286.csv"
```

## Reproduce a reference experiment

`examples/run_oqmd_svr.sh` runs the full OQMD formation-energy experiment with PCG features and SVR: all 61 eligible held-out elements, five repetitions. It then plots the result, next to the reference result if `reference_results/` was downloaded:

```bash
bash examples/run_oqmd_svr.sh "$AXY_DATA_DIR" output/oqmd_svr gpu
```

## Run other experiments

```bash
python scripts/run_ablation.py \
  --data-dir "$AXY_DATA_DIR" --out-dir output/mp_bandgap_matminer \
  --db MP --task bandgap --representation matminer --ranking elemental_properties --method RF
```

| Option | Values |
|---|---|
| `--db` | `Jarvis`, `MP`, `OQMD` |
| `--task` | `e_form` (formation energy), `bandgap` |
| `--representation` | Regressor features: `pcg` (ϕ<sub>PCG</sub>, 128 features) or `matminer` (ϕ<sub>E</sub>, 273 descriptors) |
| `--ranking` | Source ranking: `pcg`, `composition`, `embedding` or `elemental_properties` (see [Compute A<sub>XY</sub>](#compute-axy)) |
| `--method` | `RF`, `XGB`, `SVR` |
| `--test-mode` | `filtered` (default): drop test compounds containing any of the six highest-ranked sources; `unfiltered`: keep them |
| `--elements`, `--repeats`, `--device` | Held-out elements (default: all), repetitions (default: 5), `cpu` or `gpu` |

Each experiment needs the database and feature files for its dataset and representation, and the ranking file (see the data repository's README). Please download them from [data repository](https://huggingface.co/datasets/nims-spin-theory/Axy_extrapolation).

**Settings.**
- Each condition trains on exactly 10,000 compounds sampled from its pool; repetition r uses seed 42 + r.
- Test compounds contain X with atomic fraction f<sub>X</sub> > 0.4. Held-out elements need more than 100 test compounds; group-3 elements (Sc, Y, lanthanides, actinides) are not held out.
- The `pcg` and `composition` rankings come from OQMD for every dataset.

Use a new or empty `--out-dir` for every run. Held-out elements without enough training or test compounds are listed in `skips.csv`.

## Outputs

| File | Contents |
|---|---|
| `metrics.csv` | One row per held-out element, repetition and condition (`similarity`): `MAE`, `delta_MAE` (vs. the `Random` baseline), `count` (test compounds), `support_top6` and the backend |
| `summary.csv` | ΔMAE per condition, pooled over elements and repetitions, trimmed mean removing 10% from each end; `_meV` columns are in meV/atom (formation energy) or meV (bandgap) |
| `support_error.csv`, `correlations.csv` | Baseline MAE per element vs. its top-six score (mean of the six largest A<sub>XY</sub>), and their Spearman correlation |
| `predictions/<element>.csv.gz` | Per-compound predictions: `element, material_id, repeat, condition, true, prediction, ratio_X` |
| `train_ids.csv.gz`, `test_ids.csv.gz` | Compound IDs of every training set and of each element's test set |
| `source_rankings.csv`, `skips.csv`, `config.json`, `backends.json` | Rankings and excluded sources, skipped elements, run settings, model backends |

MAE is in eV/atom for formation energy and eV for bandgap.

## Summarize and plot results

Plot reference results without training any model, for example OQMD with PCG features and ranking for all three regressors:

```bash
python scripts/plot_ablation.py \
  --inputs "$AXY_DATA_DIR/reference_results/ablation_filtered" \
  --db OQMD --representations pcg --rankings pcg --methods RF XGB SVR \
  --out-dir output/reference_figures
```

`--inputs` accepts run folders, reference-result folders and individual CSV files; each file becomes one series. Select results with `--db`, `--tasks`, `--methods`, `--representations`, `--rankings`, `--test-mode` and `--elements`. The script writes `ablation.png`, `ablation.pdf`, `plot_summary.csv` and `plot_config.json`. Keep each reference CSV next to its `_config.json`, which supplies the labels.

To summarize one result file:

```bash
python scripts/summarize_ablation.py --out-dir output/summary \
  --metrics "$AXY_DATA_DIR/reference_results/ablation_filtered/pcg/ablation_OQMD_e_form_all_eSEN_layer4_SVR_base10000_testtop6_81ecfc423286.csv" \
  --support-csv "$AXY_DATA_DIR/axy/pcg/support_elements_OQMD_eSEN_layer4.csv"
```

Both scripts apply the analysis rules above: f<sub>X</sub> > 0.4, more than 100 test compounds, group 3 excluded, 10% trimmed from each end.

## Compute A<sub>XY</sub>

```bash
python scripts/compute_axy.py --data-dir "$AXY_DATA_DIR" --out-dir output/axy_pcg --db OQMD --measure pcg
```

| `--measure` | Relationship | Input |
|---|---|---|
| `pcg` | Directional alignment in ϕ<sub>PCG</sub> (A<sup>PCG</sup>) | Database and `features/pcg_layer4/` of `--db` |
| `composition` | Directional alignment in the layer-0 features (A<sup>comp</sup>) | Database and `features/composition_layer0/` of `--db` |
| `embedding` | Cosine similarity of element vectors (A<sup>embd</sup>) | `element_features/embedding_vectors.csv` |
| `elemental_properties` | Cosine similarity of 22 elemental properties (A<sup>prop</sup>) | `element_features/elemental_properties_vectors.csv` |

For directional alignment, features are standardized over all compounds. For each element, compounds are grouped into five fraction intervals of width 0.2 (at least 20 compounds each, at least three intervals). A line through the interval centroids gives the element's direction, and A<sub>XY</sub> is the cosine similarity of two directions. No property labels are used.

## Tests

```bash
pytest -q tests
```

The tests use small synthetic inputs and need no downloaded data.

## Citation

If you use this code, please cite the following article, together with the data sources listed in the data repository.

```bibtex
@article{AxyExtrapolation,
  author = {Xiao, Enda and Tadano, Terumasa},
  title  = {Materials representations organize transferable supervision for chemical extrapolation},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). The data in the [data repository](https://huggingface.co/datasets/nims-spin-theory/Axy_extrapolation) have their own licenses: CC BY 4.0, and the OMat24 License for the files containing eSEN-30M-OAM features. See the data repository's [LICENSE.md](https://huggingface.co/datasets/nims-spin-theory/Axy_extrapolation/blob/main/LICENSE.md).
