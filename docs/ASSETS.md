# Assets and Large Files Policy

## Current short-term policy
- Do not commit runtime logs/checkpoints/artifacts to Git.
- Keep model checkpoints in external storage (Drive/local output paths) or W&B artifacts.

## Ignored by default
- `runs/`
- `outputs/`
- `models/*.tar`
- `wandb/`

## Recommended workflow
1. Train/eval locally and sync metrics to W&B.
2. Keep `best.pt`/`last.pt` in configured output directory (outside source tracking).
3. If sharing models in team, publish via:
   - W&B Artifacts (preferred), or
   - GitHub Releases (explicit binary assets), not in Git history.

## Migration note
- Existing heavy files already in history are left untouched for branch safety.
- Future cleanup should use a dedicated maintenance branch/rewrite plan coordinated with the team.
