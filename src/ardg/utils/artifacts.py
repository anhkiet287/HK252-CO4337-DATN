"""Repository-local artifact registry helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ardg.config import merge_overrides
from ardg.utils.paths import project_root


def artifact_registry_dir() -> Path:
    return project_root() / "artifacts" / "latest"


def artifact_manifest_path() -> Path:
    return artifact_registry_dir() / "manifest.yaml"


def artifact_paths_doc_path() -> Path:
    return artifact_registry_dir() / "paths.md"


def ensure_artifact_registry() -> None:
    artifact_registry_dir().mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_manifest_path()
    if not manifest_path.exists():
        _write_manifest(_default_manifest())
    paths_doc = artifact_paths_doc_path()
    if not paths_doc.exists():
        paths_doc.write_text(_render_paths_doc(_default_manifest()), encoding="utf-8")


def load_artifact_manifest() -> Dict[str, Any]:
    ensure_artifact_registry()
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to manage the artifact registry.") from exc

    payload = yaml.safe_load(artifact_manifest_path().read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return _default_manifest()
    return merge_overrides(_default_manifest(), payload)


def update_artifact_manifest(updates: Dict[str, Any]) -> Dict[str, Any]:
    current = load_artifact_manifest()
    merged = merge_overrides(current, updates)
    _write_manifest(merged)
    artifact_paths_doc_path().write_text(_render_paths_doc(merged), encoding="utf-8")
    return merged


def _write_manifest(payload: Dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to manage the artifact registry.") from exc

    artifact_manifest_path().write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _default_manifest() -> Dict[str, Any]:
    return {
        "canonical": {
            "configs": {
                "default": "configs/default.yaml",
                "resnet18_train": "configs/experiments/cifar10/resnet18/baselines/erm.yaml",
                "resnet18_eval": "configs/experiments/cifar10/resnet18/eval/baseline_erm_all_attacks.yaml",
                "resnet50_train": "configs/experiments/cifar10/resnet50/baselines/erm.yaml",
                "resnet50_eval": "configs/experiments/cifar10/resnet50/eval/baseline_erm_all_attacks.yaml",
                "vit_b16_train": "configs/experiments/cifar10/vit_b16/baselines/erm.yaml",
                "vit_b16_eval": "configs/experiments/cifar10/vit_b16/eval/baseline_erm_all_attacks.yaml",
            },
            "profiles": {
                "dev_fast": "configs/profiles/dev_fast.yaml",
                "local_gpu": "configs/profiles/local_gpu.yaml",
                "colab_gpu": "configs/profiles/colab_gpu.yaml",
                "h100": "configs/profiles/h100.yaml",
            },
            "verification_scripts": {
                "wandb_policy": "scripts/qa/test_wandb_policy.sh",
                "resnet18_dev_fast": "scripts/qa/smoke_resnet18_dev_fast.sh",
                "resume_resnet18": "scripts/qa/test_resume_resnet18.sh",
                "resnet50_smoke": "scripts/qa/smoke_resnet50.sh",
                "vit_h100_smoke": "scripts/qa/smoke_vit_h100.sh",
                "preflight_local": "scripts/qa/preflight_local.sh",
                "preflight_h100": "scripts/qa/preflight_h100.sh",
                "full_verify_local": "scripts/qa/full_verify_local.sh",
            },
        },
        "latest": {
            "resnet18": {
                "train": {},
                "eval": {},
            },
            "resnet50": {
                "train": {},
                "eval": {},
            },
            "vit_b16": {
                "train": {},
                "eval": {},
            },
            "exports": {
                "json": None,
                "csv": None,
            },
            "reports": {
                "table_md": None,
                "slides": None,
            },
        },
    }


def _render_paths_doc(manifest: Dict[str, Any]) -> str:
    lines = [
        "# Latest Paths",
        "",
        "This file is the human-readable companion to `artifacts/latest/manifest.yaml`.",
        "",
        "## Canonical Configs",
    ]
    canonical_configs = manifest.get("canonical", {}).get("configs", {})
    for key, value in canonical_configs.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Canonical Profiles")
    for key, value in manifest.get("canonical", {}).get("profiles", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Verification Scripts")
    for key, value in manifest.get("canonical", {}).get("verification_scripts", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    for backbone in ("resnet18", "resnet50", "vit_b16"):
        latest = manifest.get("latest", {}).get(backbone, {})
        lines.append("")
        lines.append(f"## Latest {backbone}")
        for stage in ("train", "eval"):
            stage_payload = latest.get(stage, {})
            lines.append(f"### {stage}")
            if not stage_payload:
                lines.append("- not recorded yet")
                continue
            for key, value in stage_payload.items():
                if value in (None, "", {}):
                    continue
                lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Latest Exports")
    for key, value in manifest.get("latest", {}).get("exports", {}).items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("")
    lines.append("## Latest Reports")
    for key, value in manifest.get("latest", {}).get("reports", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines)
