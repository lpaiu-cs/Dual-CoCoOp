# Revisiting Dually-informed Conditional Context Optimization for Vision-Language Models

Korea University COSE461 Final Project  
Juneyoung Kim, Department of Computer Science, Korea University  
Jonghyeon An, Department of Computer Science, Korea University  
Revision date: March 25, 2026

## Abstract

This revised paper re-evaluates Li-CoCoOp and Dual-CoCoOp under the original CoCoOp base-to-new generalization protocol rather than the custom setting used in the earlier draft. We first reproduced the CoCoOp baseline in the current workspace and then ran matched comparisons on EuroSAT and FGVC-Aircraft with ViT-B/16, 16 shots per class, 10 epochs, 4 context tokens, and three random seeds. The reproduced CoCoOp baseline reached 73.39 harmonic mean (H) on EuroSAT and 34.00 H on FGVC-Aircraft. Li-CoCoOp improved EuroSAT to 78.36 H, but Dual-CoCoOp reached 71.44 H and did not surpass the reproduced CoCoOp baseline. On FGVC-Aircraft, neither variant improved over the reproduced baseline; Dual-CoCoOp remained close at 33.78 H, whereas Li-CoCoOp was unstable and dropped to 27.40 H. We also found several reproducibility issues in the released workspace, including missing caption assets, class-caption misalignment after base/new subsampling, and an overwritten gated-fusion path in Dual-CoCoOp. The revised evidence therefore does not support the original claim that dual conditioning consistently outperforms CoCoOp. Instead, linguistic conditioning appears dataset-dependent, and dual fusion is sensitive to caption quality and implementation details.

## 1. Introduction

Prompt learning remains one of the most practical ways to adapt CLIP to downstream tasks with limited supervision [1]. CoOp replaces the fixed hand-written prompt with learnable context vectors and improves few-shot adaptation [2]. CoCoOp further conditions those context vectors on each input image and substantially improves base-to-new generalization [3]. Motivated by the observation that many domain-specific labels remain ambiguous without extra semantic context, the original project draft proposed two extensions: Li-CoCoOp, which injects class-level linguistic descriptions, and Dual-CoCoOp, which fuses visual and linguistic conditioning through a learnable gate.

The main weakness of the original draft was experimental validity. At the time of writing, the workspace had not yet reproduced the original CoCoOp results under the official protocol, several dataset and configuration files had diverged from the reference implementation, and the caption assets required by Li-CoCoOp and Dual-CoCoOp were not included in the repository. As a result, the earlier claims of broad superiority over CoCoOp were not sufficiently supported.

This revised paper narrows the scope to a controlled and reproducible question: under the original CoCoOp base-to-new protocol, do Li-CoCoOp and Dual-CoCoOp outperform CoCoOp on representative domain-specialized datasets? We answer this question on EuroSAT [4] and FGVC-Aircraft [5], two datasets that were both central to the original motivation and locally validated end-to-end.

The revised contributions are as follows.

- We provide a controlled re-evaluation of Li-CoCoOp and Dual-CoCoOp after reproducing the CoCoOp baseline in the current workspace.
- We document the implementation and asset issues that materially affected reproducibility: missing caption JSON files, caption-label misalignment after class subsampling, an overwritten gate in Dual-CoCoOp, and missing caption-embedding initialization during evaluation.
- We show that linguistic conditioning helps on EuroSAT, but we do not find consistent gains from dual conditioning over a reproduced CoCoOp baseline, especially on FGVC-Aircraft.

## 2. Background and Method

### 2.1. CoCoOp baseline

CoCoOp extends CoOp by adding an image-conditioned bias to the learnable context tokens [3]. For an input image `x`, a small meta-network maps the frozen CLIP image feature into a context shift. The resulting prompt remains lightweight and preserves the frozen CLIP encoders, but now depends on the image instance.

### 2.2. Li-CoCoOp

Li-CoCoOp adds a second source of conditioning at the class level. Instead of relying only on the image feature, it encodes a class description with a frozen BERT encoder and maps that description through a lightweight language meta-network. The resulting language-conditioned bias is added to each learnable context token. The intuition is straightforward: if the class name alone is underspecified, a short description can resolve ambiguity before the prompt is passed to CLIP's text encoder.

### 2.3. Dual-CoCoOp

Dual-CoCoOp combines the image-conditioned bias and the language-conditioned bias with a learnable gate. Let `p_vis` be the visual bias from CoCoOp and `p_ling` be the linguistic bias from the caption branch. Dual-CoCoOp computes a gate `z = sigmoid(W[p_vis ; p_ling] + b)` and uses the fused condition `p_dual = z * p_vis + (1 - z) * p_ling`. This fused bias is then added to the shared context tokens before constructing the class prompt.

Conceptually, Dual-CoCoOp should be more expressive than either single-source variant. In practice, however, this extra flexibility increases sensitivity to implementation details, caption quality, and optimization stability.

### 2.4. Reproducibility-oriented implementation fixes

Before running the revised experiments, we had to repair several issues in the workspace. Table 1 summarizes the problems and their direct impact on evaluation.

| Issue | Effect on evaluation | Revision fix |
|---|---|---|
| Caption JSON files were absent from the repository | Li-CoCoOp and Dual-CoCoOp could not be evaluated as intended | Regenerated class captions from the appendix prompt templates and stored them locally under each dataset root |
| Caption lists were not aligned with the active base/new class subset | Language captions could be assigned to the wrong class after subsampling | Rebuilt caption lists from the active class order after subsampling |
| Dual-CoCoOp computed gated fusion and then overwrote it in the forward pass | The released implementation did not actually use the intended gate | Restored the gated fusion path in ducocoop.py |
| Caption embeddings were only prepared in before_train() | eval-only runs could fail or behave inconsistently | Moved caption-embedding preparation into the model build path so training and evaluation share the same setup |

These fixes do not change the high-level definition of the methods, but they are necessary for a fair and executable comparison.

## 3. Experimental Protocol

### 3.1. Datasets

We report revised results on EuroSAT [4] and FGVC-Aircraft [5].

- EuroSAT is a remote-sensing land-use classification benchmark where class labels benefit from semantic descriptors such as land pattern, water boundaries, or building density.
- FGVC-Aircraft is a fine-grained aircraft recognition benchmark where visually similar classes can require specialized semantic knowledge to distinguish.

These two datasets were selected because they directly represent the original motivation for adding language-level conditioning and because their CoCoOp baselines were fully reproduced in the local workspace.

### 3.2. Evaluation setting

All revised experiments follow the official CoCoOp base-to-new evaluation protocol [3].

- Backbone: CLIP ViT-B/16
- Context tokens: 4
- Shots per class: 16
- Epochs: 10
- Batch size: 1
- Optimizer: SGD, learning rate 0.002
- Scheduler: cosine decay with one warm-up epoch at `1e-5`
- Seeds: 1, 2, 3

We report base accuracy, new accuracy, and harmonic mean `H = 2 * base * new / (base + new)`. The CoCoOp numbers from the original paper are used as a historical reference, but the primary comparison in this revision is against the locally reproduced CoCoOp baseline under exactly the same runtime environment.

### 3.3. Caption generation

The repository did not provide the caption JSON files required by Li-CoCoOp and Dual-CoCoOp. To make the experiments executable, we regenerated captions from the prompt templates included in the appendix of the original project PDF. The generated caption strings were then encoded with frozen BERT-base-uncased, matching the released implementation.

This choice is an important limitation: the revised experiments evaluate Li-CoCoOp and Dual-CoCoOp with regenerated linguistic inputs rather than the original author-provided caption assets. Therefore, the results should be interpreted as a controlled reproduction in the available workspace, not as a perfect reconstruction of the original unpublished caption pipeline.

### 3.4. Hardware and software

All runs were executed locally on a machine with a single NVIDIA GeForce RTX 5090 GPU. The environment used PyTorch 2.10.0 development build with CUDA 12.8. Minor compatibility patches were required for the legacy learning-rate scheduler in the embedded Dassl code.

## 4. Results

### 4.1. Main comparison

Table 2 reports the revised mean results over three seeds. We include both the historical CoCoOp paper numbers and our reproduced CoCoOp baseline.

| Dataset | Method | Base | New | H |
|---|---|---:|---:|---:|
| EuroSAT | CoCoOp paper [3] | 87.49 | 60.04 | 71.21 |
| EuroSAT | Reproduced CoCoOp | 88.00 | 62.93 | 73.39 |
| EuroSAT | Li-CoCoOp | 88.20 | 70.50 | 78.36 |
| EuroSAT | Dual-CoCoOp | 86.87 | 60.67 | 71.44 |
| FGVC-Aircraft | CoCoOp paper [3] | 33.41 | 23.71 | 27.74 |
| FGVC-Aircraft | Reproduced CoCoOp | 35.33 | 32.77 | 34.00 |
| FGVC-Aircraft | Li-CoCoOp | 32.73 | 23.57 | 27.40 |
| FGVC-Aircraft | Dual-CoCoOp | 35.17 | 32.50 | 33.78 |

The revised conclusions are more moderate than the original draft.

- On EuroSAT, Li-CoCoOp is the strongest method in the current study. Relative to the reproduced CoCoOp baseline, it improves new-class accuracy from 62.93 to 70.50 and raises harmonic mean from 73.39 to 78.36.
- Dual-CoCoOp does not improve over the reproduced CoCoOp baseline on EuroSAT. Its harmonic mean, 71.44, is only marginally above the original paper's CoCoOp number and below both the reproduced CoCoOp baseline and Li-CoCoOp.
- On FGVC-Aircraft, neither Li-CoCoOp nor Dual-CoCoOp beats the reproduced CoCoOp baseline. Dual-CoCoOp stays close to CoCoOp, but Li-CoCoOp falls back to the level of the original paper's CoCoOp result.

The most important observation is that the reproduced CoCoOp baseline is substantially stronger than the historical CoCoOp paper result on FGVC-Aircraft. This means that comparing Li-CoCoOp or Dual-CoCoOp only against the paper-reported CoCoOp numbers would overstate their advantage in the current environment.

### 4.2. Variance across seeds

Table 3 reports the standard deviation over three seeds for the revised runs.

| Dataset | Method | Base std | New std | H std |
|---|---|---:|---:|---:|
| EuroSAT | CoCoOp | 1.14 | 16.67 | 12.42 |
| EuroSAT | Li-CoCoOp | 3.86 | 5.40 | 4.82 |
| EuroSAT | Dual-CoCoOp | 3.92 | 5.58 | 5.04 |
| FGVC-Aircraft | CoCoOp | 0.64 | 0.67 | 0.16 |
| FGVC-Aircraft | Li-CoCoOp | 2.70 | 15.05 | 13.26 |
| FGVC-Aircraft | Dual-CoCoOp | 1.53 | 0.46 | 0.55 |

Two trends stand out.

- EuroSAT is highly variable for CoCoOp on the new-class split, whereas Li-CoCoOp is both stronger and more stable. This suggests that language conditioning can regularize the prompt space when class semantics are closely tied to land-use patterns.
- Li-CoCoOp is unstable on FGVC-Aircraft, where one seed collapsed to 6.2 new-class accuracy. Dual-CoCoOp is more stable than Li-CoCoOp but still does not surpass the reproduced CoCoOp baseline on mean H.

## 5. Discussion

### 5.1. Why EuroSAT benefits from linguistic conditioning

EuroSAT labels describe semantically coherent land categories such as residential areas, annual crop fields, and permanent crop land. Short language descriptions can encode spatial layout and land-cover priors that are difficult to infer from the class name alone. In this regime, Li-CoCoOp appears to supply genuinely useful class-level bias, especially for the new classes.

### 5.2. Why FGVC-Aircraft remains difficult

FGVC-Aircraft demands fine-grained distinction among closely related aircraft families and variants. In such a setting, the quality of the captions matters more. Because the repository did not provide the original caption assets, our regenerated captions necessarily introduce noise. This issue is visible in the generated examples: some aircraft descriptions are too generic and some are simply wrong. Under these conditions, adding language conditioning does not reliably help and can even destabilize training.

### 5.3. Dual fusion is not automatically superior

The original draft argued that combining visual and linguistic cues should consistently outperform single-source conditioning. Our revised results do not support that claim. Dual-CoCoOp is competitive on both datasets and remains close to the reproduced CoCoOp baseline on FGVC-Aircraft, but it is not the best model in the current study. The extra gate and second meta-network add flexibility, but they also add failure modes. If either the linguistic input is noisy or the gate is poorly optimized, dual fusion does not guarantee improvement.

### 5.4. Reproduction should precede method claims

The strongest meta-level result of this revision is methodological rather than architectural. Several seemingly small issues in the workspace changed the experimental conclusion:

- the baseline CoCoOp setting in the repository was not the official one,
- required caption assets were missing,
- the intended gate in Dual-CoCoOp was disabled by an overwrite in the forward pass,
- and caption lists were not synchronized with base/new class subsampling.

Once these issues were corrected, the empirical story became much more nuanced. Any future claim that Li-CoCoOp or Dual-CoCoOp outperforms CoCoOp should therefore be made only after verifying a matched and reproducible baseline.

## 6. Limitations

This revised paper also has limitations.

- Only EuroSAT and FGVC-Aircraft were completed under the controlled CoCoOp reproduction protocol. We therefore make no updated claim about Food-101, DTD, or UCF101.
- The caption JSON files used in the revised experiments were regenerated locally from the appendix prompt templates rather than obtained from the original project authors.
- The current paper reports base-to-new results because that is the protocol used to validate CoCoOp reproducibility. These numbers are not directly comparable to the original project draft's all-class 8-shot 5-epoch table.
- We focus on correctness and reproducibility rather than presentation polish; the earlier draft's qualitative figures are omitted because they were not revalidated under the revised protocol.

## 7. Conclusion

This revised paper reassesses Li-CoCoOp and Dual-CoCoOp after first reproducing the CoCoOp baseline in the current workspace. The new evidence does not support the original claim that dual conditioning consistently outperforms CoCoOp. Instead, the results are dataset-dependent. Li-CoCoOp clearly helps on EuroSAT and improves both accuracy and stability on the new-class split. Dual-CoCoOp remains competitive but does not surpass the reproduced CoCoOp baseline on either dataset we validated. On FGVC-Aircraft, the reproduced CoCoOp baseline remains strongest, while Li-CoCoOp is sensitive to caption quality and seed variance.

The most defensible conclusion is therefore not that Dual-CoCoOp is universally better, but that language conditioning can help in some domains and that rigorous baseline reproduction is essential before drawing architectural conclusions. Future work should extend the validated protocol to the remaining datasets, regenerate or recover higher-quality caption assets, and study regularization strategies that make the dual gate less sensitive to noisy linguistic inputs.

## References

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.

[2] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337-2348, 2022.

[3] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. CVPR, 2022.

[4] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217-2226, 2019.

[5] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. International Conference on Image and Vision Computing New Zealand, 2013.

[6] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding, 2019.

[7] John Arevalo, Thamar Solorio, Manuel Montes y Gomez, and Fabio A. Gonzalez. Gated multimodal units for information fusion, 2017.
