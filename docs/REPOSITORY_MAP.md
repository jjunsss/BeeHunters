# Repository map

This repository separates the public reproduction package from local research
state. The separation is deliberate: Git should contain the smallest complete
path for training and inference, while large or superseded artifacts keep their
stable local paths for provenance.

## Public GitHub package

| Path | What to inspect |
|---|---|
| `README.md` | Method summary, quick start, and release scope |
| `configs/buzzspot/` | The final 12E -> 3E -> 2E training chain |
| `projects/buzzspot/` | The fixed-simplex ETF extension |
| `scripts/` | Setup, data preparation, training, inference, and release checks |
| `tests/` | Lightweight utilities and ETF geometry checks |
| `docs/REPRODUCTION.md` | Full reproduction commands and expected identities |
| `checkpoints/manifest.json` | Checkpoint sizes, SHA-256 values, and download policy |
| `data/README.md` | Dataset placement and archive identities |
| `outputs/README.md` | Supported generated-output paths |
| `THIRD_PARTY_NOTICES.md` | Upstream code, package, dataset, and weight boundaries |
| `.github/workflows/quality.yml` | Push and pull-request checks |

Run `python scripts/check_public_repo.py` to print the exact candidate file
count and fail on accidental local artifacts, large files, secrets, absolute
user paths, invalid JSON, or broken Markdown links.

## Local-only author workspace

These paths are excluded by `.gitignore` and must not be added with `git add -f`.

| Path | Local purpose |
|---|---|
| `data/raw/`, `data/extracted/`, `data/single_frame/` | Organizer archives and generated datasets |
| `checkpoints/*.pth` | Initialization and final inference weights |
| `outputs/00_final_release/` | Short symlink index to the verified final artifacts |
| `outputs/train_mmdet/`, `outputs/submissions/` | Training runs and submission packages |
| Other `outputs/` subdirectories | Historical evaluations, logs, visualizations, and sweeps |
| `legacy/` | Superseded code, configs, reports, third-party trees, and artifacts |
| `technical_report/` | Separately managed paper sources and evidence |
| `.venv*`, `.pytest_cache/` | Machine-local environments and caches |

The historical output directories remain in place because archived evidence
and long-lived local processes refer to their original paths. Use
`outputs/00_final_release/` as the human-facing entry point instead of browsing
the raw campaign directory tree.

## Where to start

1. Read `README.md` for the final method and supported commands.
2. Read `docs/REPRODUCTION.md` for exact data, checkpoint, and output identities.
3. In the author workspace, open `outputs/00_final_release/README.md` for the
   verified stage checkpoints, final submission, release checkpoint, and report
   evidence.
4. Open `legacy/README.md` only when historical experiments or excluded method
   lineages are needed.
5. Before publishing, complete `docs/GITHUB_RELEASE_CHECKLIST.md`.
