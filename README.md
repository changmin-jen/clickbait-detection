# Clickbait Detection

This repository contains a clickbait detection experiment based on precomputed
multimodal embeddings for title text, thumbnail image, speech-to-text segments,
and keyframe features.

The code is organized into reusable Python modules under `src/` and executable
experiment entrypoints under `experiments/`.

## Structure

```text
clickbait-detection/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── pooling.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   ├── train.py
│   ├── analysis.py
│   └── utils.py
├── experiments/
│   ├── pooling_experiment.py
│   └── ablation_experiment.py
├── results/
│   ├── figures/
│   └── tables/
└── README.md
```

## Usage

Run the pooling comparison experiment:

```bash
python experiments/pooling_experiment.py --pt-path /path/to/embeddings.pt
```

Run the modality ablation experiment:

```bash
python experiments/ablation_experiment.py --pt-path /path/to/embeddings.pt
```

The embedding `.pt` file is expected to contain a `samples` list with `train`,
`valid`, and `test` splits.

## Module Layout

- `src/data.py`: loads the embedding `.pt` file, builds datasets, pads variable
  length STT/keyframe sequences, and creates dataloaders.
- `src/pooling.py`: contains mean, top-k, top-k-similarity, and attention pooling
  layers.
- `src/model.py`: contains the shared branch-fusion model used by both pooling
  comparison and modality ablation experiments.
- `src/loss.py`: contains fused/branch BCE loss helpers.
- `src/metrics.py`: contains threshold search, prediction collection, and
  evaluation metrics.
- `src/train.py`: contains reusable training loops and experiment runners.
- `src/analysis.py`: contains branch-attention summary/plot utilities.
- `experiments/`: contains thin command-line entrypoints for each experiment.
