# GitHub release checklist

The repository contents are prepared for a small public release. The items
below distinguish checks that can be completed locally from decisions that
require the repository owner or challenge organizer.

## Prepared locally

- [x] Public and local-only paths are separated by `.gitignore`.
- [x] `.gitattributes` normalizes source line endings and marks model/archive
  formats as binary.
- [x] GitHub Actions checks shell syntax, Python compilation, public file scope,
  and dependency-free unit tests.
- [x] The public-scope checker rejects model files, files over 10 MiB, secrets,
  machine-specific user paths, invalid JSON, and broken local Markdown links.
- [x] Dataset, upstream-code, and checkpoint redistribution boundaries are
  documented.
- [x] Final checkpoint and submission SHA-256 identities are recorded outside
  Git history.

## Required before the first public push

- [ ] Select and add `LICENSE`. Do not infer a license from upstream
  MMDetection or from the organizer dataset terms.
- [x] Create the empty `jjunsss/BeeHunters` GitHub repository and review the
  complete initial-commit scope before pushing.
- [ ] Confirm that the organizer permits redistribution of the derived final
  checkpoint.
- [ ] Publish the sanitized inference checkpoint as a GitHub Release asset.
- [ ] Replace the null final-checkpoint URL in `checkpoints/manifest.json` and
  the placeholder URL in the README/reproduction commands.
- [ ] Run all checks below from a clean clone of the intended initial commit.

`CITATION.cff` should be added after the public repository URL, authors, title,
and preferred citation are final. `CODEOWNERS` is optional and should not be
invented without the intended GitHub usernames.

## Final local commands

```bash
bash -n scripts/*.sh
python -m compileall -q configs projects scripts tests
python scripts/check_public_repo.py
python -m unittest discover -s tests -p 'test_data_tools.py' -v
python -m unittest discover -s tests -p 'test_export_checkpoint.py' -v
python -m unittest discover -s tests -p 'test_infer.py' -v
git status --short
git remote -v
```

For the complete model-dependent test, use the recorded OpenMMLab environment:

```bash
PYTHONPATH=external/mmdetection/projects/CO-DETR:. \
  python -m unittest discover -s tests -v
```
