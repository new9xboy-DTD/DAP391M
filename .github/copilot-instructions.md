# Copilot instructions (DAP391M)

## Project snapshot
- This repo is currently mostly **data + dependencies scaffolding** for a deepfake image classifier (binary: `Fake/` vs `Real/`).
- Python code is minimal right now: `CNN + Transformer/deepfake_detection.py` only imports `numpy`.

## Data layout (important)
- Dataset root is `CNN + Transformer/Dataset/` with split folders:
  - `Train/{Fake,Real}/`
  - `Validation/{Fake,Real}/`
  - `Test/{Fake,Real}/`
- This folder structure is compatible with `torchvision.datasets.ImageFolder`-style loaders.
- Dataset source documented in `README.md` (Kaggle link + origin reference).

## Repo constraints & conventions
- Large data is intentionally excluded from agent context and should not be committed casually:
  - See `.copilotignore` (ignores `CNN + Transformer/Dataset`, `data/`, `*.csv`, `*.env`).
  - There is a `CNN + Transformer/Dataset.zip` artifact; prefer keeping datasets as external artifacts rather than adding raw images to git.
- The folder name `CNN + Transformer/` contains a space — quote paths in commands on Windows.

## Python environment
- Dependencies are pinned in `requirements.txt` (notably `torch`, `torchvision`, `timm`, `scikit-learn`, `pillow`, `tqdm`).
- If you add code that touches models, stay consistent with the pinned stack (PyTorch + torchvision + timm).

## How to run (current state)
- The only runnable script in-repo is:
  - `python "CNN + Transformer/deepfake_detection.py"`
  - Note: it currently does not train/evaluate yet.

## When making changes
- Keep new training/eval entrypoints under `CNN + Transformer/` and wire them from a clear `__main__` or CLI in the main script.
- Avoid referencing the ignored dataset paths in unit tests or static checks (the dataset may be absent on CI/machines).
- Prefer making changes that are runnable without downloading the full dataset (e.g., add a small smoke mode or graceful “dataset missing” error).
