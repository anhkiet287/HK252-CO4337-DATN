# Clean Code Rules (ARDG)

Simple, consistent rules for all contributors.

## Checklist
- Every function/class should has a Google-style docstring with `Args/Returns/Raises`.
- `scripts/` only handles CLI parsing + wiring (no heavy logic).
- All experiment settings come from config (no manual overrides at runtime).
- `wandb` logging is mandatory for every run (train/eval/attack suite).
- All config keys are used; remove or implement any unused fields.
- Deterministic runs: seeds set once and logged.
- Attacks use correct threat model and scaling/clamping.
- Checkpoints: save `last` and `best` by primary metric.

## Coding Rules
- Config-first: read settings from `cfg` only.
- Fail fast if `wandb` is disabled or missing.
- Keep functions small and single-purpose.
- Avoid magic numbers; put them in config or explain briefly.
- Use explicit typing for public APIs.

## Docstring Template (Google Style)
```python
def foo(x: int, y: str) -> bool:
    """One-line summary.

    Args:
        x: Description.
        y: Description.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When ...
    """
```
