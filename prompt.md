You are refactoring the current adversarial robustness repo into a thesis-focused experiment repo.

Context you must follow carefully:

- The current repo already has a shared training stack:
  - `scripts/train.py` is the canonical training entrypoint
  - `scripts/evaluate.py` is the canonical evaluation entrypoint
  - `src/ardg/training/trainer.py` is the shared trainer
  - model construction is already centralized
  - training objectives are already modular
  - evaluation is already unified through `attack.eval_suite`
- The current repo supports multiple modes and multiple backbones, but the thesis now needs a cleaner, narrower, more reproducible setup.
- The primary practical goal is:
  - make ResNet-18 on CIFAR-10 the clean default and best-supported path
  - keep ResNet-50 and ViT-B/16 runnable through config
  - make the repo work cleanly on local GPU, Colab GPU, and remote SSH server / H100
  - preserve resume/checkpoint behavior
  - preserve and strengthen W&B logging
- Do not rewrite the whole repo from scratch unless absolutely necessary.
- Prefer incremental, minimal-touch refactoring that keeps working pieces intact.

==================================================
A. HIGH-LEVEL REFACTOR GOAL
==================================================

Refactor the repo from a broad exploratory research framework into a thesis-oriented experiment artifact.

The final repo should:
1. Have one clean, stable default workflow centered on CIFAR-10 + ResNet-18.
2. Keep ResNet-50 as a secondary scale-up backbone.
3. Keep ViT-B/16 as an optional extension backbone for strong GPUs such as H100.
4. Separate scientific experiment logic from runtime/hardware environment settings.
5. Make experiment tracking, resume, checkpointing, and evaluation reliable enough before launching expensive runs.
6. Reduce config sprawl and move exploratory or non-final-thesis code/configs/docs into `legacy/`.

Important:
- This is a thesis repo, not a general-purpose ML framework.
- Optimize for clarity, reproducibility, and low friction.
- The default user experience must feel simple and controlled.

==================================================
B. CURRENT ARCHITECTURE YOU SHOULD PRESERVE
==================================================

Keep and reuse the following existing ideas unless there is a strong reason to change them:

1. One shared trainer loop.
2. One shared evaluation entrypoint.
3. One modular objective registry.
4. One centralized model factory.
5. One centralized attack-building interface.
6. Config-driven experiment execution.
7. Resume-from-checkpoint support.
8. Deterministic/fairness-oriented workflow.

Do not create separate train/eval scripts per backbone.
Do not create one-off duplicated pipelines for ResNet-50 or ViT.

==================================================
C. PRIMARY DESIGN DECISION
==================================================

Implement a “backbone ladder” design:

- ResNet-18:
  - primary backbone
  - fully supported
  - used for smoke tests, pilot runs, ablations, and main experiments
  - must be the cleanest and most reliable path

- ResNet-50:
  - secondary backbone
  - should reuse the same pipeline
  - only model-specific overrides should differ where necessary

- ViT-B/16:
  - optional extension backbone
  - should be runnable without re-architecting the repo
  - intended for H100 / strong GPU usage
  - should not become the default workflow

Do not add more backbones.

==================================================
D. W&B IS A HARD CONSTRAINT
==================================================

Weights & Biases logging is mandatory for canonical training/evaluation workflows.

This is a hard constraint because the user wants all runs tracked and wants W&B validated before using expensive GPUs.

Implement these rules:

1. Keep W&B configurable in YAML.
2. Keep `logging.wandb.enabled` in the schema.
3. Canonical experiment configs must use `logging.wandb.enabled: true`.
4. If `scripts/train.py` or `scripts/evaluate.py` receives a config with `logging.wandb.enabled: false`, fail immediately with a clear error.
5. Do not silently degrade train/eval when W&B is disabled.
6. Prefer practical toggling via:
   - `logging.wandb.mode: online`
   - `logging.wandb.mode: offline`
   instead of disabling W&B entirely.
7. Validate W&B as early as possible, before expensive GPU work starts.
8. Preserve and strengthen resume behavior using `wandb_run_id`.
9. Train and eval must both attach to W&B.
10. Every run must be traceable through W&B metadata.

Required W&B metadata to log:
- backbone
- dataset
- train mode / experiment type
- runtime profile
- seed
- hostname / machine identifier when available
- device / GPU name
- git commit hash if available
- resolved config
- output/run directory
- checkpoint path(s) when relevant

Required W&B behavior:
- run initialization must happen early
- resume must preserve run continuity when possible
- W&B state must be saved into run metadata and/or checkpoint sidecar
- failures in W&B setup should happen before long train loops start

Recommended W&B config shape:

logging:
  wandb:
    enabled: true
    mode: online            # online | offline
    project: ardg
    entity: ""
    group_template: "{model}/{stage}"
    add_default_tags: true
    tags: []
    run_id: null
    resume: allow

Important runtime rule:
- `enabled: false` is not acceptable for canonical train/eval flows
- `mode: offline` is acceptable for local smoke/debug if the run should still be tracked and synced later

==================================================
E. TARGET REPO STRUCTURE
==================================================

Refactor toward something close to this:

repo/
  scripts/
    train.py
    evaluate.py
    preflight_check.py
    export_results.py
    build_report_tables.py

  src/ardg/
    attacks/
    config/
    data/
    evaluation/
    experiments/
    methods/
    models/
      factory.py
      resnet.py
      vit.py
    training/
      trainer.py
      objectives/
    utils/

  configs/
    experiments/
      cifar10/
        resnet18/
          baselines/
          proposed/
          ablations/
          eval/
        resnet50/
          baselines/
          proposed/
          eval/
        vit_b16/
          baselines/
          proposed/
          eval/
    profiles/
      dev_fast.yaml
      local_gpu.yaml
      colab_gpu.yaml
      h100.yaml

  docs/
    protocol.md
    experiment_matrix.md
    run_guide.md

  legacy/
    old_methods/
    old_configs/
    exploratory/
    old_docs/

You do not need to force exactly this tree, but the separation of concerns must be clear.

==================================================
F. CONFIG REFACTOR REQUIREMENTS
==================================================

Refactor configs so they are composable and low-duplication.

Separate these concerns:
1. dataset config
2. model/backbone config
3. training method config
4. attack protocol config
5. evaluation suite config
6. runtime/hardware profile config
7. logging/tracking config

Desired behavior:
- switching ResNet-18 -> ResNet-50 -> ViT should mainly change model config
- switching local -> Colab -> H100 should mainly change runtime profile
- switching baseline -> proposed -> ablation should mainly change method/experiment config

Avoid huge monolithic YAML files with mixed responsibilities.

Keep config resolution easy to trace and debug.

==================================================
G. RUNTIME PROFILES
==================================================

Add clear runtime profiles:

1. `dev_fast.yaml`
- very short smoke settings
- tiny validation / tiny training budget
- safe local debugging
- still W&B-enabled, but can use `mode: offline`

2. `local_gpu.yaml`
- practical settings for workstation GPUs
- conservative memory usage
- mixed precision if stable
- stable local resume behavior

3. `colab_gpu.yaml`
- resilient settings for Colab
- smaller or safer batch defaults when necessary
- predictable output/checkpoint paths
- resume-friendly
- W&B setup must be easy and explicit

4. `h100.yaml`
- suitable for remote server / H100
- larger batch sizes if appropriate
- better dataloader worker defaults
- mixed precision enabled where stable
- intended for ResNet-50 and ViT as well

These profiles must only adjust environment/runtime concerns.
They must not secretly change the scientific meaning of the experiment unless explicitly documented.

==================================================
H. TRAINING PIPELINE EXPECTATIONS
==================================================

Keep `scripts/train.py` as the canonical entrypoint.

Improve it so that:
- config resolution is explicit and easy to inspect
- W&B is validated early
- logging is clear and standardized
- output directories are predictable
- resume behavior is reliable
- run metadata is easy to inspect
- there is no backbone-specific branching explosion

Need clear logging of:
- run name
- config path(s)
- resolved config
- model/backbone
- dataset
- train mode
- runtime profile
- seed
- deterministic flag
- device
- batch size
- precision
- output directory
- W&B run URL or run ID if available

The trainer should remain shared across backbones.

==================================================
I. EVALUATION PIPELINE EXPECTATIONS
==================================================

Keep `scripts/evaluate.py` as the canonical eval entrypoint.

Requirements:
- W&B logging is also mandatory here
- checkpoint auto-resolution remains supported
- deterministic eval remains supported
- JSON summary export remains supported
- quick eval vs full eval remains easy to configure
- eval runs must also log enough metadata to W&B

Do not create separate eval scripts per backbone.

==================================================
J. OUTPUT / RUN DIRECTORY STANDARDIZATION
==================================================

Standardize run outputs so they are easy to inspect and report.

Every run directory should clearly contain:
- resolved config snapshot
- checkpoints
- train logs
- eval JSON summary
- metadata
- W&B run ID
- maybe a small run manifest file

Naming should make the following obvious:
- backbone
- profile
- experiment type
- seed

Suggested naming style:
- `resnet18_local_baseline_seed42`
- `resnet18_h100_proposed_seed42`
- `resnet50_h100_uniform_seed42`
- `vit_b16_h100_proposed_seed42`

==================================================
K. CLEANUP / LEGACY POLICY
==================================================

Reduce repo sprawl.

Do the following:
- identify configs/docs/methods that are exploratory or outside the thesis-final path
- move them to `legacy/` instead of deleting immediately
- remove duplicate docs/pseudocode if not needed
- simplify README so the documented path is:
  1. ResNet-18 first
  2. ResNet-50 next
  3. ViT optional extension

The repo should no longer feel like every objective/method has equal importance.

==================================================
L. DOCUMENTATION EXPECTATIONS
==================================================

Rewrite documentation so it matches the actual intended workflow.

README should explain:
- current thesis-oriented purpose
- primary workflow: ResNet-18 on CIFAR-10
- secondary workflow: ResNet-50
- optional workflow: ViT-B/16 on strong GPU
- how to run locally
- how to run on Colab
- how to run on SSH/H100
- how to resume runs
- how W&B is set up and enforced
- where outputs are saved
- which configs are canonical

Also add:
- `docs/run_guide.md`
- `docs/protocol.md`
- `docs/experiment_matrix.md`

==================================================
M. SMOKE TESTS / PREFLIGHT
==================================================

Add or standardize lightweight commands/configs for:

- ResNet-18 smoke train
- ResNet-18 smoke eval
- ResNet-50 smoke train
- ViT smoke train
- preflight check for data / attack / normalization / model path
- W&B preflight before expensive runs

These should be safe to run locally before launching expensive jobs on Colab or H100.

==================================================
N. ACCEPTANCE CRITERIA
==================================================

At the end of the refactor, I want all of the following:

1. ResNet-18 is clearly the best-supported and cleanest path.
2. ResNet-50 and ViT are runnable through config, not through duplicated code paths.
3. Runtime profiles exist for local, Colab, and H100.
4. W&B is enforced for canonical train/eval runs.
5. W&B can be online/offline by config mode, but not fully disabled for canonical train/eval.
6. Resume + checkpoint behavior still works.
7. README and docs reflect the new intended workflow.
8. Outputs/checkpoints/run metadata are standardized.
9. Exploratory / non-final-thesis code is moved out of the main path into `legacy/`.
10. Smoke tests exist for all three backbones.

==================================================
O. HOW YOU SHOULD EXECUTE THIS WORK
==================================================

Please do this in stages, not as a blind rewrite.

Stage 1:
- audit current repo
- identify what to keep, move, rename, or archive
- propose final config layout
- propose runtime profile layout
- propose W&B hard-constraint changes

Stage 2:
- implement minimal structure/config refactor
- add runtime profiles
- standardize W&B behavior
- standardize output layout
- keep trainer/model factory shared

Stage 3:
- add/update smoke configs
- rewrite README/docs
- move unused exploratory pieces to `legacy/`

Stage 4:
- provide a concise refactor summary:
  - what changed
  - what remained shared
  - how to run ResNet-18
  - how to switch to ResNet-50
  - how to switch to ViT
  - how to use local/Colab/H100 profiles
  - how W&B behaves now

Do not over-engineer beyond this scope.