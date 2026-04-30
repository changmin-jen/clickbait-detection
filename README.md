# Clickbait Detection

This repository contains a clickbait detection experiment based on precomputed
multimodal embeddings for title text, thumbnail image, speech-to-text segments,
and keyframe features.

The original end-to-end notebook is kept in
`notebooks/clickbait_detection.ipynb`. The reusable code has been split into
Python modules under `src/` and executable experiment entrypoints under
`experiments/`.

## Structure

```text
clickbait-detection/
├── notebooks/
│   └── clickbait_detection.ipynb
├── src/
│   ├── data.py
│   ├── pooling.py
│   ├── model.py
│   ├── loss.py
│   ├── metrics.py
│   └── train.py
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
