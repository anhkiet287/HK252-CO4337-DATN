# Source Domain GroupDRO Configs

- `sd01_obj_only`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd01_obj_only.yaml` :: isolate_objective_effect
- `sd02_norm_only`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd02_norm_only.yaml` :: isolate_norm_effect
- `sd03_proc_strength_only`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd03_proc_strength_only.yaml` :: isolate_procedure_strength_effect
- `sd04_norm_x_obj_a`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd04_norm_x_obj_a.yaml` :: isolate_norm_objective_interaction
- `sd05_norm_x_obj_b`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd05_norm_x_obj_b.yaml` :: isolate_norm_objective_interaction_reverse_pair
- `sd06_mech_opt`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd06_mech_opt.yaml` :: isolate_optimization_based_mechanism_effect
- `sd07_mech_boundary`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd07_mech_boundary.yaml` :: isolate_boundary_based_mechanism_effect
- `sd08_mech_mix`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd08_mech_mix.yaml` :: test_mechanism_diversity_set
- `sd09_same_norm_with_weak`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd09_same_norm_with_weak.yaml` :: test_weak_attack_contribution_within_linf
- `sd10_balanced_mixed_strong`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd10_balanced_mixed_strong.yaml` :: test_best_balanced_mixed_set
- `sd11_no_pgd_ce_ablation`: `configs/colab/resnet50/train/source_domains/groupdro_10ep_sd11_no_pgd_ce_ablation.yaml` :: test_whether_pgd_ce_is_necessary_or_dominant
