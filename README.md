# Pixels to Predictions: DL Vision Challenge — Final Submission

QLoRA fine-tuning of `HuggingFaceTB/SmolVLM-500M-Instruct` for a
multimodal multiple-choice task (ScienceQA-style), under strict
competition constraints:

- ≤ 5M trainable parameters
- Only the SmolVLM-500M-Instruct backbone, no external data
- Offline inference required (no network at evaluation time)
- Free-tier compute only (Colab + Kaggle)

## Headline results

| Method                                | Val Acc | Public LB |
| ------------------------------------- | ------- | --------- |
| Random (1 / num_choices, averaged)    | 0.3397  | —         |
| Zero-shot SmolVLM-500M                | 0.4590  | —         |
| QLoRA fine-tuned (single-shot, T4)    | 0.7844  | 0.7887    |
| + Combined TTA (multi-res + perm), T4 | 0.7929  | 0.7988    |
| Single-shot, A100 bf16 (Colab)        | —       | 0.8229    |

The A100/T4 gap is discussed in the report: same adapter, same test
set, the only difference is `compute_dtype` (bfloat16 vs float16)
and the GPU.

## Repository

| File                              | Description                                                  |
| --------------------------------- | ------------------------------------------------------------ |
| `smolvlm_scienceqa_pipeline.ipynb`| End-to-end notebook: data loading, prompt building, 4-bit base + QLoRA training, log-likelihood scoring, single-shot evaluation, and Combined TTA (multi-resolution + choice permutation). Cell outputs from the actual training run are preserved. |
| `README.md`                       | This file.                                                   |

## Method in one paragraph

The base model is loaded in 4-bit NF4 with double quantization
(`bitsandbytes`). LoRA adapters with `r=8`, `α=16`, applied to all
seven attention/MLP projections, give 4.78M trainable parameters
(under the 5M cap). Each example is formatted as a prompt ending in
`Answer:`; we mask all label tokens except the answer letter and
train for 2 epochs at lr=2e-4 with effective batch size 8.
Inference is done by reading the next-token logits at the final
position and `argmax` over the valid letter token IDs (e.g.
`' A'`, `' B'`, ...) — note the leading space, which matches what
the model actually emits after `Answer:`. Combined TTA averages
logits across three image resolutions (384, 448, 512) and across
four random choice-position permutations.

## Reproducing the submission

### Hardware

- **Training:** one Colab A100 (40GB or 80GB), bfloat16. Roughly
  76 minutes for 2 epochs.
- **Inference (offline):** Kaggle T4 with Internet OFF, float16.
  Roughly 60 minutes for the full test set with Combined TTA and
  `do_image_splitting=False` (required to fit the 9-hour commit
  budget on T4).

### Steps

1. Open `smolvlm_scienceqa_pipeline.ipynb` in Colab with an A100.
2. Run sections 1–9 to reproduce the LoRA adapter (`lora_best/`).
3. The same notebook also produces a `submission.csv` from this
   adapter directly on Colab. For the offline path, package the
   adapter and the SmolVLM-500M base into Kaggle datasets, then
   run an offline Kaggle notebook that mirrors the inference
   functions in the Colab notebook (with `Internet = OFF`).

A full LaTeX report with ablations and error analysis is provided
separately.

## Hyperparameters at a glance

| Setting                      | Value                       |
| ---------------------------- | --------------------------- |
| Quantization                 | NF4 + double quant          |
| LoRA rank `r`                | 8                           |
| LoRA `α`                     | 16                          |
| LoRA dropout                 | 0.05                        |
| LoRA target modules          | q, k, v, o, gate, up, down  |
| Trainable parameters         | 4{,}784{,}128 (4.78M)       |
| Epochs                       | 2                           |
| Effective batch size         | 8 (2 × grad accum 4)        |
| Learning rate                | 2e-4                        |
| LR schedule                  | cosine, η_min = 0.1·LR      |
| Warmup                       | 3% of total steps           |
| Gradient clipping            | 1.0                         |
| Image resolution (training)  | 384 × 384                   |
| Image splitting (training)   | enabled                     |
| Image splitting (T4 inference) | disabled (time budget)    |

## Notes

- All competition rules were followed: only the provided data and
  the specified backbone were used; no external pretrained models
  or datasets; submission generated offline.
- Total training and inference compute fits within free-tier
  Colab + Kaggle resources.

## Author

Peifeng Yao (`py2330@nyu.edu`), New York University.
