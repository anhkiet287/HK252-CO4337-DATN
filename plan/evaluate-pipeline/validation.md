# Validation

## Commands Run
```bash
# Syntax sanity for evaluate path
python3 -m py_compile \
  scripts/evaluate.py \
  src/ardg/evaluation/evaluator.py \
  src/ardg/attacks/attack_suite.py
```

## Recommended Runtime Validation Commands
```bash
# 1) Evaluate a trained checkpoint (clean + PGD20, and AA if enabled in config)
PYTHONPATH=src python3 scripts/evaluate.py \
  --config configs/local/resnet50/at/multi_attack_erm.yaml \
  --checkpoint outputs/resnet50_local_multi_attack_erm/best.pt \
  --splits val,test \
  2>&1 | tee plan/evaluate-pipeline/evidence/02_attack_suite/eval_multi_attack.log

# 2) Save qualitative attack visuals for report evidence
PYTHONPATH=src python3 scripts/attack_visual_check.py \
  --config configs/local/resnet50/at/multi_attack_erm.yaml \
  --checkpoint outputs/resnet50_local_multi_attack_erm/best.pt \
  --split test --attack-source eval_pgd20 --attack from_source \
  --num-samples 8 --strict-eps \
  --output-dir plan/evaluate-pipeline/evidence/05_visualization/multi_attack_eval_pgd20
```

## Results
- Evaluation architecture documented and linked to actual script/class flow.

## Gaps / Risks
- Need run artifacts from at least one ERM checkpoint and one adversarial checkpoint.
- AutoAttack path requires optional dependency and may increase runtime significantly.
