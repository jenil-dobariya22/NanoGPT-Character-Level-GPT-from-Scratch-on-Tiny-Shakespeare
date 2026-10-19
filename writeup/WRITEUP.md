# Character-Level GPT on Tiny Shakespeare: Training, Baselines and Head Ablation

*Numbers come from the single completed run in `nanogpt.ipynb` (preset `tiny`, seed 1337, CPU). Outputs are in `results/` and `figures/`.*

## 1. Summary

I implemented a decoder-only Transformer from scratch in PyTorch (following the architecture of Karpathy's "Let's build GPT") and trained a 0.82M-parameter, 4-layer, 4-head character-level model on Tiny Shakespeare for 3,000 steps on CPU. It reaches a validation loss of **1.626 nats/char** (2.346 bits/char, perplexity 5.08), compared with 2.482 for a bigram baseline and 4.174 for a uniform guess, a 0.86 nat (35%) improvement over the bigram. Four sanity checks (parameter count, initial loss, causal mask, overfitting one batch) pass. Zero-ablating individual attention heads shows that layer-0 head 0 is by far the most important (+0.480 validation loss), followed by layer-0 heads 3 and 2; no head in layers 1-3 changes the loss by more than +0.021. These are results for one small model, one seed and one dataset.

## 2. Motivation

The aim was to understand a Transformer language model end to end: build it, check that it is correct, compare it with a trivial baseline, look inside it, and test what the attention heads contribute. A bigram baseline shows whether the model uses context at all; head ablation gives a simple causal measure of component importance (as in Michel et al., 2019).

## 3. Data

Tiny Shakespeare (Karpathy's char-rnn copy): 1,115,394 characters with a 65-symbol vocabulary (letters, digits, punctuation, space, newline). The first 90% (1,003,854 characters) is used for training and the last 10% (111,540) for validation. The split is by position, not shuffled, because the text is one continuous stream; the validation data is never used to update weights.

## 4. Methods

**Model.** Token and learned position embeddings, then 4 pre-LayerNorm Transformer blocks (multi-head causal self-attention with 1/sqrt(head_size) scaling, a 4x ReLU feed-forward network, residual connections), a final LayerNorm and a linear output layer. d_model = 128, 4 heads of size 32, context 64, dropout 0.0, weights initialised from N(0, 0.02). Total 816,705 parameters (checked against an expected count in the notebook).

**Training.** AdamW, constant learning rate 1e-3, batch 32, 3,000 steps, evaluation every 300 steps. Random training windows; evaluation always uses the same fixed batches (the generator is re-seeded to 999), so models and checkpoints are directly comparable. The checkpoint with the best validation loss is kept (here the final step). All randomness is seeded (1337).

**Baselines.** A uniform guess (loss ln 65 = 4.174) and a bigram model (a 65x65 table predicting the next character from the current one), trained 3,000 steps with AdamW at LR 1e-2. Its initial loss (4.72) is above ln 65 because `nn.Embedding` is initialised from N(0, 1); the GPT uses std 0.02 and starts near ln 65 (4.21).

**Sanity checks.** (1) Parameter count equals the architecture's expected value (816,705). (2) Logits have shape (32, 64, 65) and the initial loss is within 0.3 of ln 65 (4.213). (3) Causality: changing the character at position t leaves all logits before t unchanged (and does change those from t on). (4) Overfitting one batch: loss falls from 4.21 to 0.03 in 200 steps.

**Head ablation.** For each of the 16 heads, the head's output is replaced by zeros (zero-ablation) and the validation loss is re-measured on the same fixed batches; the original method is restored afterwards and the model is asserted to be unchanged. The reported quantity is the increase over the unablated validation loss.

**Attention heatmaps.** Attention weights are recomputed for a short passage ("First Citizen:\nBefore we proceed any further, hear me speak.") for layer 0 heads 0 and 1 and layer 3 head 0.

## 5. Results

### 5.1 Language modelling

| Model | Val loss (nats/char) | Bits/char | Perplexity |
|---|---|---|---|
| Uniform guess | 4.174 | 6.022 | 65.0 |
| Bigram | 2.482 | 3.581 | 11.96 |
| GPT-tiny | **1.626** | **2.346** | **5.08** |

Training log for the GPT (train / validation): step 0: 4.218 / 4.214; step 300: 2.093 / 2.139; step 900: 1.662 / 1.810; step 1,500: 1.533 / 1.711; step 2,100: 1.475 / 1.654; step 3,000: 1.419 / 1.626. Validation loss is flat between steps 2,400 and 2,700 (1.638, 1.638) and lowest at the final step. The train/validation gap is about 0.21, so the model appears limited by size and training length more than by overfitting. The full run took about 24 minutes on CPU. The bigram converges to 2.482 (train 2.460) with almost no real words in its samples.

### 5.2 Samples

The GPT reproduces the play format (capitalised speaker names with colons, line breaks, punctuation) and many real words, but also invents non-words and does not produce coherent sentences. Low temperature (0.5) gives more conservative, repetitive text; high temperature (1.3) gives more errors. Samples are saved in `results/sample_tiny.txt`.

### 5.3 Head ablation

Baseline validation loss 1.6259.

| layer | head | val loss | delta |
|---|---|---|---|
| 0 | 0 | 2.1061 | +0.4802 |
| 0 | 3 | 1.7988 | +0.1729 |
| 0 | 2 | 1.7207 | +0.0948 |
| 0 | 1 | 1.6512 | +0.0253 |
| 1 | 3 | 1.6472 | +0.0213 |
| 2 | 1 | 1.6449 | +0.0190 |
| 1 | 2 | 1.6413 | +0.0154 |
| 2 | 0 | 1.6403 | +0.0144 |
| 2 | 3 | 1.6398 | +0.0139 |
| 2 | 2 | 1.6372 | +0.0113 |

(Top 10 of 16 heads; all 16 are in `results/head_ablation.csv`.) Zero-ablating layer 0 head 0 raises the loss by 0.480, nearly three times the next head. All three heads of layer 0 other than head 1 matter noticeably; every head in layers 1-3 changes the loss by at most 0.021. Under zero-ablation, then, first-layer heads matter most.

### 5.4 Attention heatmaps

`figures/attention_L0_H0.png`, `attention_L0_H1.png` and `attention_L3_H0.png` show for each query position how much attention goes to each earlier position; the upper triangle is empty because of the causal mask. I have not made claims about what specific heads compute from these plots.

## 6. Interpretation

The GPT clearly uses context: it is far better than the bigram (which sees one character) and its perplexity of about 5 means that on average it is as uncertain as a choice among 5 characters. The ablation suggests that, in this model, much of the useful attention computation sits in layer 0, with later heads contributing little individually.

This does not show that later heads are redundant or that layer 0 "does the real work". Two reasons: zero-ablation can move activations off their normal distribution, which can inflate the damage (mean-ablation is a gentler alternative that was not run), and layer-0 heads feed every later layer, so a large effect there partly reflects position in the network rather than the head's own function.

## 7. Limitations

- **One run, one seed (1337).** No seed-to-seed variation was measured; head indices have no fixed meaning across training runs, so the specific "layer 0 head 0" result may not transfer to another run.
- **Validation estimate.** Validation loss is the mean over 100 fixed random overlapping windows (204,800 characters, more than the 111,540 validation characters), not an exact pass over the validation set. The checkpoint is selected on the same batches that are reported, so the number can be slightly optimistic (negligible here, because the best checkpoint is the last step). There is no separate test set.
- **Small scale.** 0.82M parameters, constant learning rate, 3,000 steps, no learning-rate schedule; the loss was still decreasing. Larger presets (`medium`, `full`) exist in the notebook but were not run.
- **Zero-ablation only**, single heads only (no pairs, no mean-ablation, no per-position analysis), and no interpretation of what individual heads compute.
- **The depth sweep (Section 14 of the notebook) was not run** (`RUN_SWEEPS = False`); no result in this report depends on it. Because it uses a short training budget it would mainly measure early-training speed, not the final benefit of depth.
- Character-level modelling of one author's text; nothing here says how the findings carry over to subword models or other data.

## 8. Reproducibility

Run `nanogpt.ipynb` from the repository root, top to bottom. Seeds: 1337 for the model and data stream, 999 for the evaluation batches. The notebook downloads Tiny Shakespeare to `data/input.txt`. The GPT run used about 24 minutes on CPU (torch 2.14.0+cpu); exact loss values can vary slightly across hardware and library versions. Files produced: `results/loss_tiny.csv`, `results/summary.csv`, `results/head_ablation.csv`, `results/sample_tiny.txt`, `results/gpt_tiny_best.pt`, and figures in `figures/`.

## References

- Karpathy, *Let's build GPT: from scratch, in code, spelled out* and the nanoGPT repository
- Vaswani et al., *Attention Is All You Need* (2017), arXiv:1706.03762
- Radford et al., *Language Models are Unsupervised Multitask Learners* (2019)
- Karpathy, char-rnn and the Tiny Shakespeare dataset (2015)
- Michel et al., *Are Sixteen Heads Really Better than One?* (2019), arXiv:1905.10650