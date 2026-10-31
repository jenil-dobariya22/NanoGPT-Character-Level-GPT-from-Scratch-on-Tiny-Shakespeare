# Character-Level GPT from Scratch: Training, Interpretability, and Ablation on Tiny Shakespeare

A decoder-only, character-level Transformer implemented from scratch in PyTorch and trained to predict the next character in the Tiny Shakespeare corpus. The notebook covers the complete workflow: data preparation, a bigram baseline, Transformer implementation, correctness checks, training, text generation, attention visualisation, and per-head zero-ablation.

The project is intended as a hands-on study of Transformer internals and component-level evaluation—not as a claim that attention-head ablation alone provides a complete explanation of model behaviour.

## Key results

The reported run uses the `tiny` preset on CPU, with seed `1337` and 3,000 training steps.

| Model | Validation loss (nats/character) | Bits/character | Perplexity |
|---|---:|---:|---:|
| Uniform prediction | 4.174 | 6.022 | 65.00 |
| Bigram baseline | 2.482 | 3.581 | 11.96 |
| **GPT-tiny** | **1.626** | **2.346** | **5.08** |

The GPT-tiny validation loss is approximately **35% lower** than the bigram baseline (a reduction of about 0.86 nats per character). The model learns useful character-level and play-format patterns, although its generated passages can still contain malformed words and incoherent sentences.

### Attention-head zero-ablation

Each attention head was evaluated by replacing its output with zeros and measuring validation loss again. The unablated model's validation loss was **1.6259**.

| Layer | Head | Validation loss after ablation | Change from baseline |
|---:|---:|---:|---:|
| 0 | 0 | 2.1061 | **+0.4802** |
| 0 | 3 | 1.7988 | +0.1729 |
| 0 | 2 | 1.7207 | +0.0948 |
| 0 | 1 | 1.6512 | +0.0253 |
| 1 | 3 | 1.6472 | +0.0213 |

Layer 0, head 0 has the largest measured effect in this trained model. The result is specific to this checkpoint and this zero-ablation procedure; it does not establish that later-layer heads are generally redundant.

## Dataset and preprocessing

- **Dataset:** Tiny Shakespeare, downloaded by the notebook from [Karpathy's character-RNN dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).
- **Corpus size:** 1,115,394 characters.
- **Vocabulary:** 65 unique characters, encoded as integer IDs using a character-to-index and index-to-character mapping.
- **Task:** Given a context of preceding characters, predict the next character at every position.
- **Split:** The first 90% of the text is used for training and the final 10% for validation. The split follows the original text order and is not shuffled; validation characters are not used for weight updates.
- **Context length:** 64 characters in the reported `tiny` run.

The notebook downloads the corpus automatically. A working internet connection is therefore needed on the first run, unless `data/input.txt` is supplied manually.

## Model architecture

The GPT-tiny configuration contains **816,705 trainable parameters** and uses:

| Component | Configuration |
|---|---|
| Architecture | Decoder-only Transformer / character-level GPT |
| Transformer blocks | 4 |
| Attention heads per block | 4 |
| Embedding dimension | 128 |
| Context length | 64 characters |
| Feed-forward hidden dimension | 4 × embedding dimension (512) |
| Attention | Scaled, causal multi-head self-attention |
| Normalisation | Pre-LayerNorm within each block, plus final LayerNorm |
| Residual connections | Around attention and feed-forward sublayers |
| Activation | ReLU in the feed-forward network |
| Dropout | 0.0 for the `tiny` preset |
| Embeddings | Learned token and positional embeddings |
| Output | Linear projection to the 65-character vocabulary |

The causal attention mask prevents a position from attending to future characters. Attention scores are scaled by the square root of the head dimension before softmax.

## Training configuration

The reported GPT-tiny run uses:

| Setting | Value |
|---|---:|
| Preset | `tiny` |
| Device | CPU |
| Random seed | 1337 |
| Batch size | 32 |
| Block size | 64 |
| Embedding dimension | 128 |
| Number of heads | 4 |
| Number of layers | 4 |
| Learning rate | 0.001 |
| Optimizer | AdamW |
| Training steps | 3,000 |
| Evaluation interval | 300 steps |
| Evaluation batches per split | 100 |
| Mixed precision | Disabled |

The bigram baseline is trained for 3,000 steps using AdamW with a learning rate of `0.01`. During GPT training, the notebook records train and validation loss, saves the checkpoint with the best measured validation loss, reloads that checkpoint, and then runs the final evaluation.

The notebook also defines `medium` and `full` presets. With `PRESET = None`, it selects `medium` when CUDA is available and `tiny` on CPU. To request the configuration that produced the results above on any supported device, explicitly set `PRESET = "tiny"` in the switches cell.

## Evaluation protocol and implementation checks

Validation loss is estimated using 100 fixed batches generated from a re-seeded random generator. This keeps evaluation batches comparable across checkpoints and models. For the `tiny` preset, the estimate samples 204,800 character positions from random, potentially overlapping validation windows. It is therefore an estimate on sampled windows, not a single exact pass over every validation character.

The notebook runs four implementation checks before full training:

1. **Parameter count:** verifies that the model has the expected 816,705 parameters.
2. **Output shape and initial loss:** checks the `(batch, context, vocabulary)` logits shape and confirms that initial loss is close to `ln(65)`.
3. **Causality:** changes a character at a later position and verifies that earlier-position predictions remain unchanged.
4. **Single-batch overfit:** trains repeatedly on one fixed batch and confirms that its loss falls close to zero.

All four checks pass in the recorded run.

## Interpretability and analysis

### Attention visualisation

The notebook extracts and plots attention patterns for selected heads. Each heatmap shows the attention weights from query positions to attended-to key positions. The upper triangle is empty because the causal mask blocks access to future positions.

### Head ablation procedure

For each of the 16 attention heads, the notebook temporarily replaces that head's output with zeros, measures validation loss using the same fixed evaluation batches, and records the loss difference relative to the unablated model. The original head implementation is restored after each measurement, and an assertion verifies that the model's evaluation loss returns to its original value after the ablation loop.

**Interpretation caveats:**

- Zero-ablation can move activations outside their normal distribution; mean-ablation would be a different intervention.
- Earlier-layer heads can affect representations processed by every later layer, so a large first-layer effect may partly reflect its position in the network.
- The experiment evaluates one trained model and one seed. Head indices do not have a fixed functional meaning across different training runs.
- A small loss change under single-head ablation does not prove that a head is unused or that its computation is unimportant in combination with other heads.

### Optional depth sweep

A multi-seed depth sweep is implemented in the notebook but is **not run** in the reported experiment (`RUN_SWEEPS = False`). No conclusions about the relative benefits of different model depths are claimed from this project.

## Text generation

The trained model generates text one character at a time, optionally from a prompt. The sampling temperature controls the sharpness of the next-character distribution:

- Lower temperature produces more conservative and often more repetitive text.
- Higher temperature increases variation but also tends to produce more errors.

The generated passages reproduce some surface conventions of Shakespearean dialogue—speaker labels, punctuation, and line breaks—but are not consistently grammatical or semantically coherent.

## Repository contents

A typical project directory after running the notebook looks like this:

```text
.
├── nanogpt.ipynb                 # Complete implementation, training, and analysis
├── README.md                     # Project documentation
├── data/
│   └── input.txt                 # Downloaded Tiny Shakespeare corpus
├── figures/
│   ├── loss_curves_tiny.png
│   └── attention_L*_H*.png
└── results/
    ├── gpt_tiny_best.pt          # Best-validation GPT checkpoint
    ├── loss_tiny.csv             # GPT training history
    ├── summary.csv               # Uniform, bigram, and GPT comparison
    ├── head_ablation.csv         # Per-head ablation results
    └── sample_tiny.txt           # Generated sample
```

The `data/`, `figures/`, and `results/` folders are created by the notebook as needed. Files are generated when their corresponding cells are executed.

## How to run

1. Clone or download this repository and open `nanogpt.ipynb` in Jupyter Notebook or JupyterLab.
2. Install Python and the project dependencies:

   ```bash
   pip install torch numpy pandas matplotlib jupyter
   ```

   Install a PyTorch build appropriate for your operating system and CUDA setup if you intend to use a GPU.

3. In the notebook's **Setup / Switches** section, set:

   ```python
   PRESET = "tiny"       # use the reported configuration
   DRYRUN = False        # run the full training procedure
   USE_AMP = False       # mixed precision is not used in the reported run
   RUN_ABLATION = True   # run the 16-head ablation experiment
   RUN_SWEEPS = False    # optional depth sweep; not part of reported results
   ```

4. Run the notebook from top to bottom. The dataset is downloaded automatically if `data/input.txt` is not already present.
5. Review the loss curves, generated sample, attention heatmaps, and CSV files in `figures/` and `results/`.

For a quick execution check, set `DRYRUN = True`. This reduces the number of training steps and is only intended to test that the notebook cells execute; it does not reproduce the reported metrics.

## Limitations

- This is a small character-level language model trained on one literary corpus; the results should not be generalised to modern language models.
- The reported GPT result is from one seed and a 3,000-step CPU run. Longer training, different seeds, and larger presets were not evaluated in the reported results.
- Validation loss is estimated from fixed sampled windows, and the best checkpoint is selected using that same validation estimate; the resulting reported value may be slightly optimistic.
- Zero-ablation is a useful diagnostic intervention, but it is not a complete causal account of an attention head's role.
- The optional depth sweep was not executed, so the notebook does not support claims about the best model depth.

## Acknowledgements

The notebook follows the educational architecture of Andrej Karpathy's ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY), with its own implementation checks, baseline comparison, training protocol, attention visualisation, and ablation analysis. The Tiny Shakespeare text is sourced from the [char-rnn repository](https://github.com/karpathy/char-rnn).
