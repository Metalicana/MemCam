# Experiment and Handoff Guide

For editable figures and saved results, start with the
[paper workspace](../paper/README.md). For upstream installation and inference,
see the [repository README](../README.md).

## Running Experiments

- [Cluster runbook](cluster_runbook.md) and [Slurm notes](../slurm/README.md)
- [Complete metric grid](budget_metric_grid.md)
- [Query-latency experiment](query_latency_experiment.md)
- [Pose/appearance ablation](complete_pose_appearance_ablation.md)
- [Weight ablation](keepsake_weight_ablation.md)
- [Native round-trip evaluation](keepsake_native_roundtrip.md)
- [Matched RI/KEEPSAKE FVD comparison](fvd_ri_keepsake_30.md)
- [Random-five quality analysis](random5_quality.md)
- [CUT3R and WorldScore](cut3r_worldscore_metrics.md)

## Diagnostics and Figures

- [Retention/selection analysis](retention_selection_analysis_plan.md)
- [Diagnostic experiment summary](memcam_diagnostic_experiments_summary.md)
- [Memory strips](memory_strips.md)
- [60-second profiling](prof_60s_analysis.md)

## Cross-System Handoffs

- [WorldMem final evaluation](worldmem_final_evaluation_handoff.md)
- [WorldMem retention/selection gaps](worldmem_retention_selection_handoff.md)
- [WorldMem retrieval deterioration](worldmem_retrieval_deterioration_handoff.md)
- [DFoT/FramePack transfer study](cecsl_dfot_framepack_handoff.md)

## Background and Historical Notes

Planning briefs, meeting notes, and mechanism-hunt documents remain at their
existing paths. Treat their proposed experiments and provisional results as
historical context, not completion records. The previous paper-status README
is preserved in [archive/paper_status_2026-09-06.md](archive/paper_status_2026-09-06.md).

The cleanup did not rename scripts in `paper/`, `utils/`, or `slurm/`, or change
remote result locations. Explicit local input paths now use `paper/configs/`
and `paper/results/`; historical export provenance keeps its original paths.
