# HyperPEFT-LoRA: Synthesizing Unseen User Personas

Code for the paper *Personas on Demand: Hypernetwork Synthesis of Unseen Users*.
A hypernetwork emits per-user LoRA adapters for a **frozen Pythia-1.4B**
backbone. The study asks whether the same mechanism that rebuilds known users can synthesize
behaviorally consistent **unseen** users by sampling new points in the profile-vector space, and
characterizes where that synthesis holds and where it breaks down.

Synthetic users are drawn at three distances from the real-user manifold:
**in-hull** (deployment), **near-hull** (boundary robustness), and **far-from-hull** (a deliberate
off-manifold `kappa`-pushout). Fidelity is measured with per-measure signed Hedges' *g*, the Persona
Stability Index (PSI), the Persona Compositionality Index (PCI), and a multi-turn drift audit.

## Repository layout
- **`data_scripts/`** — dataset loading (`hypernetwork_dataset_10000.py`), the profile-vector /
  feature builder (`hypernetwork_feature_builder_10000.py`), the hypernetwork + injection
  architecture (`hypernetwork_structure_10000.py`), and the unseen-persona samplers
  (`sample_far_from_hull_extrapolation.py`, `sample_midpoint_baseline.py`).
- **`training_scripts/`** — the hypernetwork training loop (`train_hyperlora.py`), the vanilla-LoRA
  baseline, forum generation/inference (`build_hyperlora_forum.py`), the inference-time
  residual-stream steering (`extract_arditi_directions.py`, `arditi_*`), and evaluation
  (`signed_hedges_g.py`, `compute_pci.py`, `score_*`).
- **`figures/`** — scripts that regenerate the paper's figures and numeric artifacts.
- **`yaml_scripts/`** — Kubernetes job specs and the orchestration shell script used to run
  training and evaluation on a GPU cluster.

## Setup
- Python 3.9+, with PyTorch, `transformers`, `peft`, `pandas`, `numpy`, `scikit-learn`, and
  `matplotlib`.
- **Paths are placeholders.** Scripts use `/workspace/...` for the project root and `/data/...` for
  bulk data and model caches. Set these to your own environment (edit the `*_ROOT` / `*_DIR`
  constants near the top of each script, or the volume mounts in `yaml_scripts/`).
- Backbone: Pythia-1.4B (EleutherAI), kept frozen. Scoring uses GoEmotions
  (`SamLowe/roberta-base-go_emotions`), SST-2 (`distilbert-base-uncased-finetuned-sst-2-english`),
  and VADER, each run as released.

## Data
The [`data/`](data/) directory holds the released artifacts: the per-author profile vectors
(`profile_vectors_10000.parquet`, the 18 model-input features), the thread-level train/val/test
split (`thread_split_10000.parquet`), and the cohort and axis labels (`labels/`). See
[`data/README.md`](data/README.md) for schemas.

Profile vectors are computed from the public Pushshift Reddit archive (2010 to 2016). Usernames are
irreversibly hashed (SHA-256) before processing; no raw post text, account names, or person-tied
weights are redistributed. Every identifier in the released files is an integer with no bridge back
to an account name. See the paper's Data and Ethical Considerations sections.

## Citation
If you use this code, please cite the paper and the HyperPEFTNet preprint it builds on. Full
citation details are added on publication.

## License
Released under the MIT License. See [`LICENSE`](LICENSE).
