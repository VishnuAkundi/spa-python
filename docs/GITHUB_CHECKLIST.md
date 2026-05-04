# GitHub Checklist

Use this before pushing or making a release for the lab.

## Before First Push

```bash
cd /path/to/spa_python
git init
git status --short
```

Confirm that audio, outputs, reports, caches, and `.venv/` are not listed.

Then:

```bash
source .venv/bin/activate
pytest
python -m pip check
```

## Files That Should Be Tracked

- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `run_spa.py`
- `spa_app/`
- `spa_core/`
- `tests/`
- `tools/`
- `docs/`
- `.gitignore`

## Files That Should Not Be Tracked

- `.venv/`
- `.matplotlib/`
- `.pytest_cache/`
- `__pycache__/`
- `audio/`
- `outputs/`
- `spa_outputs/`
- `reports/`
- participant audio/media files
- generated CSV/XLSX outputs

## Suggested First Commit

```bash
git add README.md requirements.txt pyproject.toml run_spa.py spa_app spa_core tests tools docs .gitignore
git commit -m "Prepare SPA Python app for lab use"
```

Choose a lab-appropriate license before making the repository public. If the
repository is private, document who is allowed to access participant data.
