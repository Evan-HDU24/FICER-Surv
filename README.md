# FICER-Surv

**Function-Informed Core-Preserving Evidence Refinement for Cancer Survival Prediction**

**Built with PriorLabs-TabPFN**

**Implementation:** TabPFN-FICER-Surv · **Backbone:** frozen TabPFN-3.

FICER-Surv predicts cancer survival risk from gene-level molecular profiles. Group-aware survival screening (GSS) selects outcome-relevant features within six functional groups. Core-anchored graph refinement (CAGR) preserves strong screening evidence and refines the remaining selection. Time-conditioned survival inference (TCSI) uses the selected measurements and observable patient–time labels to condition a frozen TabPFN-3 classifier.

![Motivation](assets/motivation.png)

## Abstract

Accurate cancer survival prediction benefits from tumor molecular profiles, including mutation status, copy-number changes, and RNA expression. However, high dimensionality and right censoring challenge prognostic learning from small cohorts. To reduce the need for cohort-specific training, in-context learning reuses pretrained predictive priors through labeled patient contexts without updating backbone weights. Effective transfer therefore requires compact, outcome-relevant molecular evidence and observable, censoring-compatible context labels. We propose FICER-Surv, a function-informed, core-preserving framework that refines molecular contexts for pretrained survival inference. Group-aware survival screening allocates functional-group budgets and uses Elastic-Net Cox to select evidence supported by censored outcomes. To incorporate molecular relationships while preserving strong prognostic signals, core-anchored graph refinement protects high-priority variables and propagates importance over within-group correlation graphs to revise non-core inputs. Time-conditioned survival inference combines selected measurements with observable patient–time cumulative-event labels, excluding unknown censored states; frozen TabPFN uses these contexts to estimate cumulative-event probabilities, survival curves, and risk. Experiments on four cohorts from The Cancer Genome Atlas show improved average prognostic discrimination over the evaluated baselines under matched feature budgets.

## Framework

![FICER-Surv framework](assets/framework.png)

## Installation

The implementation is tested with Python 3.12, PyTorch 2.7.1 with CUDA 12.8, and TabPFN 8.2.0.

```bash
conda create -n ficer-surv python=3.12 -y
conda activate ficer-surv
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

### TabPFN access and first run

This implementation fixes `tabpfn==8.2.0` and explicitly selects **TabPFN-3**. Keep this version and checkpoint to reproduce the reported results.

1. With `checkpoint` set to `null`, the official TabPFN package obtains its classifier weights on first use. Its download flow requires a Prior Labs account and acceptance of the applicable model license. Follow the browser sign-in prompt or visit [Prior Labs](https://ux.priorlabs.ai).
2. For a terminal without a browser, accept the license through your own account's License tab and set `TABPFN_TOKEN` in the environment before running. Keep tokens private; do not put them in `options.json`, source files, or commits.
3. With a legitimately obtained local checkpoint, provide its path directly. No repeated weight download is needed:

```bash
python main.py --dataset ucec --fold 0 --checkpoint /path/to/tabpfn-v3-classifier-v3_default.ckpt
```

Weights are **not included** in this repository. The local-checkpoint route supports offline inference after dependencies and weights have been obtained. This project uses the locally installed TabPFN package rather than the hosted TabPFN Client API. See the [version-specific official setup instructions](https://github.com/PriorLabs/TabPFN/blob/v8.2.0/README.md#installation--setup) for account and download troubleshooting.

### Third-party licenses

The installed TabPFN 8.2.0 code is covered by the **Prior Labs License, version 1.2 (Apache 2.0 with an additional attribution provision)**. TabPFN-3 weights have a separate **TABPFN-3 Non-Commercial License**. In particular, availability of the Python package does not remove the conditions on model weights or outputs. The weight license covers qualifying non-commercial research and evaluation subject to its terms; commercial use, production deployment, and hosted services require separate authorization from Prior Labs.

Unmodified copies of both licenses are included under [`licenses/`](licenses/). Attribution, the precise use of upstream implementations, data provenance, and references are documented in [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). These notices do not grant permission on behalf of upstream rights holders, replace their terms, or represent approval by Prior Labs. This repository is a research implementation and does not provide a clinical service.

## Run

Run one held-out fold:

```bash
python main.py --dataset ucec --fold 0
```

Run all five folds for a cohort:

```bash
python main.py --dataset ucec --fold all
python main.py --dataset luad --fold all
python main.py --dataset blca --fold all
python main.py --dataset gbmlgg --fold all
```

`main.py` is the entry point; configuration is provided in `options.json`.

## CSV data

The cohort data and functional gene lists are adopted from the resources released with [MCAT](https://github.com/mahmoodlab/MCAT), with the original patient data originating from [The Cancer Genome Atlas](https://www.cancer.gov/ccg/research/genome-sequencing/tcga).

| File | Cohort | Patients |
|---|---|---:|
| `ucec.csv` | Uterine Corpus Endometrial Carcinoma | 480 |
| `luad.csv` | Lung Adenocarcinoma | 453 |
| `blca.csv` | Bladder Urothelial Carcinoma | 373 |
| `gbmlgg.csv` | Glioblastoma & Lower Grade Glioma | 568 |

## Files

```text
FICER-Surv/
├── main.py             # Command-line entry point
├── options.json        # Configuration
├── requirements.txt
├── README.md
├── ficer_surv/
│   ├── __init__.py
│   ├── model.py        # FICER-Surv model composition and interface
│   ├── gss.py          # Functional groups, budgets, and EN-Cox screening
│   ├── cagr.py         # Correlation graphs, importance propagation, and core protection
│   ├── tcsi.py         # Survival contexts, TabPFN inference, and risk recovery
│   └── data.py         # CSV loading, validation, and patient folds
├── data/
├── assets/
└── licenses/
```

The model interface is available through `from ficer_surv import FICERSurv`.

## Acknowledgments

We thank the Prior Labs team and TabPFN contributors for making their implementation and pretrained models available. FICER-Surv uses their implementation code through the installed `tabpfn` package, including `TabPFNClassifier` and `ModelVersion.V3`. This repository does not vendor the TabPFN source tree or distribute its pretrained weights; the frozen backbone remains the upstream model.

The cumulative-event classification formulation used in TCSI follows Kim, Lai, and Zhang's [Tabular Foundation Models Can Do Survival Analysis](https://arxiv.org/abs/2601.22259). Molecular and survival data originate from [The Cancer Genome Atlas](https://www.cancer.gov/ccg/research/genome-sequencing/tcga), and `data/signatures.csv` is retained from the [MCAT](https://github.com/mahmoodlab/MCAT) functional gene lists. We acknowledge these methodological and data contributions separately from the FICER-Surv evidence-construction stages.

## References

- Grinsztajn et al. [TabPFN-3: Technical Report](https://arxiv.org/abs/2605.13986), 2026.
- Hollmann et al. [Accurate predictions on small data with a tabular foundation model](https://doi.org/10.1038/s41586-024-08328-6), Nature, 2025.
- Kim, Lai, and Zhang. [Tabular Foundation Models Can Do Survival Analysis](https://arxiv.org/abs/2601.22259), 2026.
- Chen et al. [Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html), ICCV, 2021.
