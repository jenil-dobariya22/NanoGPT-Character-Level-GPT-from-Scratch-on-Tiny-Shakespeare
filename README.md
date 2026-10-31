# NanoGPT: Character-Level GPT from Scratch on Tiny Shakespeare

A decoder-only Transformer (GPT) written from scratch in PyTorch and trained to predict the next character on Tiny Shakespeare (1.1M characters, 65-symbol vocabulary).

The project goes beyond training. It verifies the implementation with sanity checks, compares against a bigram baseline, inspects attention patterns, and measures how much each attention head matters by ablating it.

## Main results

Preset `tiny` (0.82M parameters, 3,000 steps, CPU, seed 1337):

| Model | Validation loss (nats/char) | Bits/char | Perplexity |
|---|---|---|---|
| Uniform guess | 4.174 | 6.022 | 65.0 |
| Bigram baseline | 2.482 | 3.581 | 11.96 |
| **GPT-tiny (4 layers, 4 heads, d=128)** | **1.626** | **2.346** | **5.08** |

- The GPT lowers validation loss by 0.86 nats (**35%**) relative to the bigram baseline.
- **Head ablation:** zero-ablating **layer 0, head 0** raises validation loss by **+0.48**, the largest effect, nearly 3x the next largest (layer 0 head 3, +0.17). No head in layers 1-3 changes the loss by more than +0.021.

## What is inside

| Part | Description |
|---|---|
| Model | Token + position embeddings, causal multi-head self-attention, pre-LayerNorm blocks, feed-forward layers, residual connections |
| Baseline | Bigram model (next character from the current character only) |
| Sanity checks | Parameter count, output shape and initial loss, causality, overfit-one-batch |
| Analysis | Training curves, text generation (with temperature), attention heatmaps, per-head zero-ablation |
| Optional | Depth sweep over number of layers (off by default, not part of the reported results) |

## Sanity checks

All four checks pass before training:

1. **Parameter count** matches the expected architecture (816,705 for `tiny`).
2. **Initial loss** is close to ln(65) = 4.17, so the loss and initialisation are correct.
3. **Causality:** changing a character never changes earlier predictions, so the attention mask is correct.
4. **Overfit one batch:** loss drops from 4.21 to 0.03 on a single repeated batch, so the model and training step work.

## Model configurations

| Preset | Layers | Heads | n_embd | Block size | Parameters |
|---|---|---|---|---|---|
| `tiny` (reported) | 4 | 4 | 128 | 64 | 816,705 |
| `medium` | 6 | 4 | 256 | 128 | 4,800,577 |
| `full` | 6 | 6 | 384 | 256 | 10,788,929 |

The preset is chosen automatically from the available hardware (GPU gives `medium`, CPU gives `tiny`). All reported numbers use `tiny`.

## How to run

1. Install the requirements:

   ```bash
   pip install torch numpy pandas matplotlib jupyter
   ```

2. Open the notebook and run all cells:

   ```bash
   jupyter notebook nanogpt.ipynb
   ```

3. The dataset (Tiny Shakespeare) is downloaded automatically to `data/input.txt` on the first run.

Useful switches (Section 1.2 of the notebook):

- `PRESET`: force `"tiny"`, `"medium"` or `"full"`.
- `DRYRUN = True`: a very short run (about 100 steps) to check that every cell works.
- `RUN_ABLATION`: head ablation (on by default).
- `RUN_SWEEPS`: optional depth sweep (off by default, about 20 minutes on CPU).

## Reproducibility

All randomness is seeded (`SEED = 1337`). Evaluation always uses the same fixed batches, so numbers are directly comparable across models and checkpoints. The best-validation checkpoint is saved and reloaded before reporting.

## Output files

Running the notebook creates:

- `results/`: loss history (`loss_tiny.csv`), `summary.csv`, `head_ablation.csv`, generated text, best checkpoint
- `figures/`: loss curves and attention heatmaps

## Limitations

- The validation loss is estimated on sampled, overlapping windows (204,800 characters per evaluation for `tiny`), not an exact pass over the 111,540 validation characters.
- The best checkpoint is chosen on the same fixed validation batches that are reported, so the validation number can be slightly optimistic. Here the best checkpoint is the final step, so the effect is negligible.
- Zero-ablation can push activations off their normal distribution (mean-ablation is gentler). Heads in early layers influence all later layers, so a large layer-0 effect partly reflects its position and not only the head's own function.
- The ablation result is for one trained model, and head indices have no fixed meaning across training runs.
- The model is small and was trained for a short budget. The generated text has the format of the play and many real words, but also non-words and incoherent sentences.
- Longer training or the `medium` and `full` presets would likely lower the loss further, but this was not tested.

## Credits

Based on the architecture from Andrej Karpathy's "Let's build GPT" (nanoGPT lineage). The implementation, sanity checks, baseline and analyses were written for this notebook.
