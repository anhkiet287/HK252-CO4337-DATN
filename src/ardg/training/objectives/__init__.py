"""Objective registry for Trainer."""

from __future__ import annotations

from typing import Any, Dict

from ardg.training.objectives.base import Objective
from ardg.training.objectives.erm import ERM
from ardg.training.objectives.pgd_at import PGDAT
from ardg.training.objectives.rex import REx
from ardg.training.objectives.groupdro import GroupDRO
from ardg.training.objectives.groupdro_plus import GroupDROPlus
from ardg.training.objectives.multi_attack_erm import MultiAttackERM


def build_objective(cfg: Dict[str, Any], model: Any) -> Objective:
    """Instantiate objective based on cfg.train.mode."""
    mode = cfg["train"]["mode"].lower()
    if mode in {"erm", "clean"}:
        return ERM(cfg)
    if mode == "pgd_at":
        return PGDAT(cfg, model)
    if mode in {"multi_attack_erm", "multi-attack-erm", "multi_attack"}:
        return MultiAttackERM(cfg, model)
    if mode == "rex":
        return REx(cfg, model)
    if mode in {"groupdro", "group_dro"}:
        return GroupDRO(cfg, model)
    if mode in {"groupdro_plus", "groupdro++"}:
        return GroupDROPlus(cfg, model)
    raise ValueError(f"Unsupported train.mode for Trainer: {cfg['train']['mode']}")
