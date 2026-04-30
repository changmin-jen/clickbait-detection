# Clickbait Detection

Multimodal clickbait detection experiments using precomputed embeddings from:

- title text
- thumbnail image
- speech-to-text segments
- video keyframes

The project compares sequence pooling strategies and runs modality ablation
experiments over four modality-pair branches.

## Project Structure

```text
clickbait-detection/
|-- experiments/
|   |-- pooling_experiment.py
|   `-- ablation_experiment.py
|-- src/
|   |-- __init__.py
|   |-- analysis.py
|   |-- data.py
|   |-- loss.py
|   |-- metrics.py
|   |-- model.py
|   |-- pooling.py
|   |-- train.py
|   `-- utils.py
|-- results/
|   |-- figures/
|   `-- tables/
`-- README.md
```

## Data Format

The experiment expects a PyTorch `.pt` file with this structure:

```python
{
    "samples": [
        {
            "id": "...",
            "split": "train",  # train, valid, or test
            "label_id": 0,     # 0 or 1
            "title_emb": Tensor[D],
            "thumb_emb": Tensor[D],
            "stt_embs": Tensor[N_stt, D],
            "kf_embs": Tensor[N_keyframes, D],
        },
        ...
    ]
}
```

The default model assumes `D = 768`.

## Experiments

### Pooling Comparison

Compares `mean`, `topk`, `topk_sim`, and `attention` pooling for variable-length
STT/keyframe embeddings.

```bash
python experiments/pooling_experiment.py --pt-path /path/to/embeddings.pt
```

Run selected pooling methods:

```bash
python experiments/pooling_experiment.py \
  --pt-path /path/to/embeddings.pt \
  --pooling mean attention \
  --epochs 30 \
  --batch-size 32
```

### Modality Ablation

Evaluates which modality-pair branches contribute most to classification.

Branches:

- `tk`: title + keyframe
- `ts`: title + STT
- `thk`: thumbnail + keyframe
- `ths`: thumbnail + STT

Run all ablations:

```bash
python experiments/ablation_experiment.py --pt-path /path/to/embeddings.pt
```

Run selected ablations:

```bash
python experiments/ablation_experiment.py \
  --pt-path /path/to/embeddings.pt \
  --experiments All "w/o TS" "w/o ThK"
```

## Source Modules

- `src/data.py`: dataset, padding, collate function, dataloader creation
- `src/pooling.py`: mean, top-k, top-k similarity, and attention pooling
- `src/model.py`: shared branch-fusion model for both experiment types
- `src/loss.py`: branch and fused BCE loss helpers
- `src/metrics.py`: accuracy, precision, recall, F1, AUC, threshold search
- `src/train.py`: training loop, early stopping, experiment runners
- `src/analysis.py`: branch attention summary and plot utilities
- `src/utils.py`: device selection, batch device movement, seed setup

## Outputs

Pooling experiment checkpoints are saved to `results/models/` by default.
Figures and tables can be saved under:

- `results/figures/`
- `results/tables/`

## Notes

The original notebook has been refactored into Python modules so experiments can
be rerun from the command line and each component can be maintained separately.
