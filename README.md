# Order, Keys, and Yes/No: Auditing Option Bias in a Non-Autoregressive Decision Model

Code, raw model outputs and figures for the paper by **Sagar Maheshwari** (Independent Researcher).

📄 Paper: [`paper/laya-bias-audit.pdf`](paper/laya-bias-audit.pdf) · arXiv link: *coming soon*

## What this is

[Laya](https://huggingface.co/convaiinnovations/laya) is an open, non-autoregressive *decision model*: given an input and typed questions (multi-class `choice`, yes/no `noul`), it returns calibrated probabilities in one encoder forward pass. This project audits whether its answers depend on **how a question is presented** rather than on the input, across its English and multilingual checkpoints, four datasets (AG News, DAIR Emotion, SST-2, BoolQ) and ~35,000 queries.

## Key findings

- **Order mostly doesn't matter, until the task gets hard.** On AG News both checkpoints give the same answer under all 24 option orderings for 94–96% of items. On six-way Emotion, consistency drops to 74% / 62%, and the multilingual checkpoint shows a significant **recency bias** (23.6% of choices go to the last slot vs. 17.4% expected).
- **Option keys can override their descriptions.** When a key names a different class than its description, the model follows the key in 80.5–92.0% of AG News cases.
- **Yes/no wording is fragile.** "Is this review positive?" and "Is this review negative?" contradict each other on 91–94% of SST-2 reviews. Under one wording the English checkpoint's ranking is **inverted** (AUROC 0.352), which is hidden by the library's default 4-decimal output rounding.
- **Fixes.** Zero-label contextual calibration helps fixed-template sentiment questions (up to +25 points) but never helps on BoolQ and hurts when probes are per question. Platt scaling on 200 labels never significantly reduced accuracy (paired McNemar tests, Holm-corrected).

## Repository layout

```
laya_audit_full.py   # runs every experiment on both checkpoints; caches raw outputs; prints tables; draws figures
paired_stats.py      # paired McNemar + bootstrap tests with Holm correction, from cached outputs (no GPU needed)
results/             # raw outputs (JSONL), results_summary.json, paired_stats.{md,json}
figures/             # the three figures in the paper
paper/               # compiled paper (PDF)
```

## Reproducing

**Full run (GPU, ~25 min on one T4).** On Kaggle: enable *GPU T4 x2* and *Internet*, then

```bash
pip install -r requirements.txt
python laya_audit_full.py        # writes /kaggle/working/laya_audit/
```

Raw outputs are cached, so an interrupted run resumes where it stopped. The script disables Laya's 4-decimal output rounding (display only; model computation is unchanged) because some findings concern probabilities below 1e-4. Results in the paper use **laya 0.3.22**; newer releases or retrained checkpoints may differ.

**Statistics only (CPU, seconds).** Recompute the paired tests from the included raw outputs:

```bash
pip install numpy scipy scikit-learn
python paired_stats.py results
```

## Data format

Each `results/*.jsonl` line is one item (or one item × condition) with the dataset index, gold label, the condition (ordering, key scheme or framing) and the model's full-precision probabilities. **No dataset text is included**; the original datasets keep their own licenses and can be loaded from the Hugging Face Hub by index (`fancyzhx/ag_news`, `dair-ai/emotion`, `stanfordnlp/sst2`, `google/boolq`).

## Citation

```bibtex
@misc{maheshwari2026laya,
  title  = {Order, Keys, and Yes/No: Auditing Option Bias in a Non-Autoregressive Decision Model},
  author = {Maheshwari, Sagar},
  year   = {2026},
  note   = {arXiv preprint},
  url    = {https://github.com/maheshwari2000/laya-bias-audit}
}
```

## Acknowledgements and disclosure

Laya is developed by Nandakishor M / ConvAI Innovations and contributors ([GitHub](https://github.com/NandhaKishorM/laya)). This work used Claude (Anthropic) extensively for literature search, experiment design, code, analysis and drafting; see the paper's *Use of AI Assistants* section.

## License

Code: MIT (see `LICENSE`). Raw outputs and figures: CC BY 4.0.
