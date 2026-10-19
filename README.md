# SAE Features vs Attention Explanations: Faithfulness on GPT-2-small

A small research project comparing **attention-based** and **SAE-feature-based** token explanations for GPT-2-small's few-shot sentiment predictions on SST-2. Faithfulness is measured with ERASER-style comprehensiveness and sufficiency, with random, raw-gradient, positional and attention-gradient controls. A held-out SAE-feature ablation (with a non-SAE direction baseline) tests whether the features matter causally.

> **Status:** pipeline complete and run end-to-end on CPU (N = 200). Items marked `TODO` need manual work (Neuronpedia feature inspection, final case-study choice). See [`writeup/WRITEUP.md`](writeup/WRITEUP.md) for the full report.

## TL;DR

- **Task:** GPT-2-small, 4-shot prompt, margin = logit(" positive") - logit(" negative"), on SST-2 validation sentences the model classifies correctly (100 per class).
- **Explanations:** last-position attention; attention x gradient; gradient x activation through the `gpt2-small-res-jb` SAE at layer 8; raw gradient x activation (no SAE); last-k-tokens baseline; random ranking.
- **Test:** top-k token removal (comprehensiveness) and retention (sufficiency), k = 1, 2, 3, 5, 8, with 95% bootstrap CIs and paired tests (Bonferroni over 140 tests).
- **SAE-specific:** features ranked on a DEV half, ablated on a held-out TEST half, against matched-random, activation-only and bottom-K controls, plus a difference-in-means direction baseline.

## Main results (N = 200, mean [95% bootstrap CI])

**Comprehensiveness** (drop in gold-label margin when the top-k tokens are removed; higher = more faithful)

| method | k=1 | k=3 | k=5 |
|---|---|---|---|
| attention | 0.028 [0.013, 0.043] | 0.053 [0.014, 0.091] | 0.095 [0.049, 0.139] |
| attention x gradient | 0.087 [0.062, 0.113] | 0.194 [0.154, 0.231] | 0.234 [0.189, 0.276] |
| SAE (layer 8) | -0.038 [-0.056, -0.020] | 0.039 [0.005, 0.074] | 0.080 [0.040, 0.120] |
| raw grad x act | 0.028 [0.007, 0.050] | 0.073 [0.040, 0.107] | 0.107 [0.071, 0.142] |
| last-k tokens | 0.032 [0.017, 0.047] | 0.040 [0.024, 0.055] | 0.073 [0.054, 0.090] |
| random | 0.003 [-0.012, 0.017] | 0.010 [-0.016, 0.036] | 0.039 [0.008, 0.069] |

**Sufficiency** (drop in margin when only the top-k tokens are kept; lower = more faithful)

| method | k=1 | k=3 | k=5 |
|---|---|---|---|
| attention | 0.230 [0.210, 0.249] | 0.182 [0.152, 0.212] | 0.145 [0.109, 0.181] |
| attention x gradient | 0.170 [0.146, 0.195] | 0.075 [0.046, 0.102] | -0.001 [-0.033, 0.029] |
| SAE (layer 8) | 0.204 [0.180, 0.228] | 0.142 [0.115, 0.169] | 0.087 [0.055, 0.118] |
| raw grad x act | 0.183 [0.154, 0.212] | 0.122 [0.088, 0.154] | 0.078 [0.039, 0.116] |
| last-k tokens | 0.229 [0.209, 0.248] | 0.186 [0.151, 0.220] | 0.149 [0.104, 0.195] |
| random | 0.239 [0.216, 0.263] | 0.220 [0.191, 0.248] | 0.187 [0.157, 0.216] |

![faithfulness curves](figures/faithfulness_curves.png)

### What the numbers support

- **Attention x gradient** had the best mean value on both metrics at every k. Against the SAE score it was significantly better on comprehensiveness at every k (dz between -0.69 and -0.81 for SAE minus attention x gradient, Bonferroni p < 0.001) and on sufficiency from k=2 on (not significant at k=1). It was also clearly better than random (dz up to 1.02 on comprehensiveness). Direct tests against plain attention and raw grad x act were not part of the test set. It is signed and label-aware like the gradient methods, so part of its advantage may be structural.
- **SAE token scores vs attention:** on comprehensiveness, the SAE score was *worse* at k=1 (dz = -0.43) and not distinguishable from attention at k = 2-8 after correction. At k=1 its comprehensiveness was below random (mean difference -0.041, Bonferroni p = 0.004).
- **SAE token scores vs random on sufficiency:** the SAE score was better than random at every k (for example k=5: mean difference -0.100, dz = -0.47, Bonferroni p < 0.001), but not clearly better than raw grad x act or last-k.
- **SAE and raw grad x act are not similar here.** Mean Spearman correlation was -0.155 and `resid_ratio` (mean |raw - SAE| / mean |raw|) was 5.57. The identity SAE score = (x_hat - b_dec) . g is exact, but it implies nothing about correlation: the leftover term (b_dec + error) . g is large relative to the raw score on this setup.

### Held-out feature ablation (sentence-span scope, K = number of features ablated)

Drop in gold-label margin on the **held-out** half (51 NEG / 49 POS sentences). Mean margin of the evaluation set is 0.245.

| class | K | top-K (DEV rank) | top-K by activation only | bottom-K | matched random (mean) |
|---|---|---|---|---|---|
| NEG | 1 | 0.010 | -0.001 | -0.010 | 0.000 |
| NEG | 10 | 0.029 | -0.004 | -0.021 | -0.010 |
| NEG | 50 | 0.048 | 0.011 | -0.028 | -0.014 |
| POS | 1 | 0.034 | 0.005 | -0.003 | 0.002 |
| POS | 10 | 0.072 | 0.044 | -0.017 | 0.009 |
| POS | 50 | 0.101 | 0.082 | -0.036 | -0.002 |

Top-K features reduced the held-out margin more than matched random features (paired effect size dz between 0.78 and 2.92 across all rows). Bottom-K features moved the margin the opposite way, as expected. For POS sentences, ablating features chosen only by activation size (no gradient) reproduced most of the effect, and the top features fire in nearly every sentence, so they look like dense, generic features. **No feature has been inspected on Neuronpedia yet (`TODO`), so none is called a "sentiment feature".**

### Non-SAE direction baseline

One difference-in-means direction (rank-1), compared with 15 label-shuffled null directions. Two projections are reported: the plain projection (removes the whole component along the direction) and a mean-centred projection (removes only the part that varies around its DEV mean).

| class | plain drop | null mean +/- SD | dz vs null | centred drop | centred null mean |
|---|---|---|---|---|---|
| POS | 0.123 | -0.006 +/- 0.025 | 2.16 | 0.055 | -0.001 |
| NEG | -0.017 | -0.002 +/- 0.015 | -0.38 | 0.037 | 0.001 |

With the plain projection, one direction removed more margin than 50 SAE features on POS sentences (0.123 vs 0.101) and slightly increased the NEG margin. With the centred projection the NEG anomaly disappears and the POS effect falls to 0.055, below the top-50 SAE features (POS 0.101, NEG 0.048). Much of the plain-projection effect therefore came from removing the shared offset along the direction. The comparison is not matched in size (1 direction vs 50 features).

### SAE reconstruction fidelity at layer 8

Replacing the layer-8 residual by the SAE reconstruction (error term dropped) left the mean margin almost unchanged (0.228 -> 0.220) but only **68%** of these originally-correct predictions stayed correct. Leaving the BOS position untouched and replacing only the other positions gave the same result (68% still correct, mean margin 0.221), so the loss is not caused by BOS. Margins are small, so conclusions that rely on the reconstruction alone are weak; the main experiments keep the error term constant.

## What this shows, and what it does not

**Shows:** on GPT-2-small, SST-2, layer 8, this prompt and N = 200, how these explanation methods rank on removal-based faithfulness, and that SAE features chosen by gradient x activation have a causal effect on held-out sentences beyond matched random features.

**Does not show:** that SAEs are better or worse than attention in general; that any SAE feature is "the" sentiment feature; anything about other models, layers, tasks or prompts. Removal-based metrics use out-of-distribution inputs; the classifier is weak (SST-2 accuracy 0.658, positive-class accuracy 0.392); normalised metrics use only the 97 / 200 sentences with margin gap above 0.25. Part of the gap between attention x gradient and the SAE / raw scores may come from where they are computed (SAE and raw scores at layer 8 only; attention x gradient over all 12 layers). The SAE library recommends loading the model without weight processing; this run used the default processing (not tested).

## Setup details

| Item | Value |
|---|---|
| Model | GPT-2-small (TransformerLens `HookedTransformer`) |
| SAE | SAELens release `gpt2-small-res-jb`, id `blocks.8.hook_resid_pre` (24,576 features) |
| Data | GLUE SST-2 validation, correctly predicted, 10-40 sentence tokens, class balanced (100 per class; 337 NEG and 155 POS candidates) |
| Baseline | few-shot accuracy 0.658, predicted-positive rate 0.232, true-positive rate 0.509 |
| k values | 1, 2, 3, 5, 8 |
| Random baseline | 3 random rankings per sentence per k, averaged |
| Feature ablation | K in {1, 5, 10, 20, 50}; 5 random sets per K; DEV = even rows, TEST = odd rows |
| Seed | 0 |
| Hardware | CPU only |

**Prompt** (4-shot; sentences written in SST-2 style; the evaluation sentence replaces the text after the final `Review:`):

```
Review: a gorgeous , witty and deeply moving film .
Sentiment: positive

Review: a dull , lifeless mess with no redeeming qualities .
Sentiment: negative

Review: the performances are wonderful and the story is engaging .
Sentiment: positive

Review: it is boring , predictable and far too long .
Sentiment: negative

Review: <sentence>
Sentiment:
```

**Library versions:** torch 2.14.0, transformer-lens 3.9.0, sae-lens 6.54.4, datasets 5.0.1, pandas 3.0.6, scipy 1.17.1, numpy 2.4.6, matplotlib 3.11.2.

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── SAE.ipynb                      # full pipeline, run top to bottom
├── data/selected_examples.csv     # the exact sentences evaluated
├── figures/                       # faithfulness curves, example plots, feature ablation
├── results/                       # per-example CSV, summary, paired tests, ablation, run_info.json
└── writeup/WRITEUP.md             # full report
```

## Quickstart

```bash
git clone https://github.com/<your-username>/sae-vs-attention-faithfulness.git
cd sae-vs-attention-faithfulness
pip install -r requirements.txt
jupyter notebook SAE.ipynb
```

Run from the repo root, top to bottom. In the config cell, `QUICK = True` is a ~40-sentence dry run (outputs go to `results/quick/` and `figures/quick/`); `QUICK = False` is the full run. The full run on CPU takes hours; a CUDA GPU makes it much faster. The main loop checkpoints every 25 sentences and resumes automatically.

## References

- Jain & Wallace, *Attention is not Explanation* (2019), arXiv:1902.10186
- Wiegreffe & Pinter, *Attention is not not Explanation* (2019), arXiv:1908.04626
- Pruthi et al., *Learning to Deceive with Attention-Based Explanations* (ACL 2020), arXiv:1909.07913
- Serrano & Smith, *Is Attention Interpretable?* (ACL 2019), arXiv:1906.03731
- DeYoung et al., *ERASER: A Benchmark to Evaluate Rationalized NLP Models* (ACL 2020), arXiv:1911.03429
- Cunningham et al., *Sparse Autoencoders Find Highly Interpretable Features in Language Models* (2023), arXiv:2309.08600
- Bricken et al., *Towards Monosemanticity* (Anthropic, 2023)
- Hooker et al., *A Benchmark for Interpretability Methods in Deep Neural Networks* (2019), arXiv:1806.10758
- Tigges et al., *Linear Representations of Sentiment in Large Language Models* (2023), arXiv:2310.15154
- TransformerLens, SAELens, Neuronpedia

## License

MIT. See `LICENSE`.