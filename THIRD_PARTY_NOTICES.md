# Third-party implementations, models, and data

**Built with PriorLabs-TabPFN**

## TabPFN implementation and pretrained model

FICER-Surv uses third-party implementation code through the installed [`tabpfn==8.2.0`](https://github.com/PriorLabs/TabPFN/tree/v8.2.0) dependency. Specifically, `ficer_surv/tcsi.py` imports `TabPFNClassifier` and `ModelVersion`, instantiates the official TabPFN-3 classifier, and invokes its `fit` and `predict_proba` interfaces. In this usage, `fit` conditions the frozen predictor on the constructed context; this project does not optimize the backbone weights. The local molecular-selection stages and survival wrapper are provided as source in `ficer_surv/`.

The official TabPFN source tree is not copied into this repository, and no pretrained checkpoint is bundled. Acknowledging an installed implementation dependency is distinct from claiming that its source files were authored here or copied into these local modules.

| Resource | Applicable upstream license | Included copy |
|---|---|---|
| TabPFN 8.2.0 implementation | Prior Labs License, version 1.2, December 2025: Apache 2.0 with an additional attribution provision | [Code license](licenses/TabPFN-8.2.0-LICENSE.txt) |
| TabPFN-3 pretrained model | TABPFN-3 License v1.0, revised March 24, 2026: non-commercial and non-production terms | [Model license](licenses/TabPFN-3-LICENSE.txt) |

The code license's additional provision includes prominent attribution requirements and a naming condition for AI models created or improved using the covered resources. Its complete wording is retained in the included license. The public inference implementation is identified as **TabPFN-FICER-Surv**, retaining the TabPFN prefix; **FICER-Surv** remains the paper and framework name. Its unchanged pretrained inference backbone is explicitly identified as **TabPFN-3**. No newly trained, fine-tuned, or distilled TabPFN checkpoint is released here.

The model license governs qualifying research and evaluation as well as restrictions on commercial use, production, outputs, redistribution, and hosted services. A local research demo does not confer permission to deploy a hosted prediction service. Obtain any additional permission directly from Prior Labs where the intended use requires it. License acceptance and model access are the responsibility of each user; the FICER-Surv authors do not grant or sublicense Prior Labs' rights.

The model attribution notice is reproduced in [`NOTICE`](NOTICE). This repository does not imply endorsement or validation by Prior Labs. The complete upstream terms take precedence over the summaries on this page. Copies of the licenses were retrieved from the following official locations on October 26, 2026:

- [TabPFN v8.2.0 code license](https://github.com/PriorLabs/TabPFN/blob/v8.2.0/LICENSE)
- [TabPFN-3 model license](https://huggingface.co/Prior-Labs/tabpfn_3/blob/main/LICENSE)

## Survival formulation

The cumulative-event classification formulation used by TCSI follows:

Da In Kim, Wei Siang Lai, and Kelly W. Zhang. **Tabular Foundation Models Can Do Survival Analysis.** 2026. [arXiv:2601.22259](https://arxiv.org/abs/2601.22259).

The authors' reference implementation is available at [TabularFM4Survival](https://github.com/dain20000222/TabularFM4Survival).

The local wrapper constructs patient–time inputs, omits unobservable censored labels, invokes the frozen classifier, and recovers monotone survival estimates and risk scores. This attribution concerns the methodological formulation; it does not claim a new pretrained TabPFN architecture or ownership of the upstream inference implementation.

## Functional gene signatures

`data/signatures.csv` is an unmodified copy of the functional gene lists used by the [MCAT repository](https://github.com/mahmoodlab/MCAT). The local grouping procedure assigns overlapping genes according to the listed group priority. No MCAT neural-network implementation is bundled.

Richard J. Chen et al. **Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images.** ICCV, 2021. [Official paper](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Multimodal_Co-Attention_Transformer_for_Survival_Prediction_in_Gigapixel_Whole_Slide_ICCV_2021_paper.html).

The upstream repository's GPL-3.0 license is reproduced without alteration in [`licenses/MCAT-LICENSE.txt`](licenses/MCAT-LICENSE.txt) for the accompanying third-party resource. This notice does not assert ownership of the gene lists or change their applicable terms.

## Cohort data and other dependencies

The cohort CSVs derive from [The Cancer Genome Atlas](https://www.cancer.gov/ccg/research/genome-sequencing/tcga). They contain molecular candidate measurements, follow-up outcomes, patient identifiers, and the evaluation-fold assignments described in the README. Source data terms and attribution requirements remain applicable; no new ownership of TCGA data is claimed.

NumPy, pandas, SciPy, scikit-learn, scikit-survival, PyTorch, and threadpoolctl are installed dependencies, not vendored source. Their respective licenses remain applicable to their distributions. This attribution document does not relicense these dependencies, model weights, data, or any other third-party resource.
