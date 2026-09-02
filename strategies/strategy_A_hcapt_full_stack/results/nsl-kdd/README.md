# Results — `nsl-kdd`

## Provenance note (2026-08-12)

Until 2026-08-12 this folder also contained 15 files copied verbatim from the
Phase-A NSL-KDD run in `05-Implementation/results/`:

    ablation_matrix.json         baselines_results.json      ensemble_selflearn_results.json
    final_comparison.csv         final_evaluation.json       gan_augmented_ensemble.json
    gan_comparison.json          master_blockchain.json      master_pipeline_summary.json
    per_class_breakdown.csv      selflearn_events.json       v2_blockchain.json
    v2_pipeline_summary.json     zero_trust_results.json     zt_audit_summary.json

They were byte-identical in every dataset folder and carried **NSL-KDD numbers**
(accuracy 0.7618; class supports Normal = 9,711, DoS = 7,460 — the KDDTest+
counts), regardless of which dataset folder they sat in. Anyone reading
`final_evaluation.json` from this folder would have read NSL-KDD results.

They are **retained here** because this *is* the NSL-KDD folder, so the numbers are
correct for this dataset. They were removed from the ton-iot-telemetry, edge-iiotset
and dapt-2020 folders, where they were misleading.

The originals are untouched in `05-Implementation/results/`.

**The authoritative numbers for this dataset are in `REPORT.md` and
`ablation_table.csv`**, which are genuinely per-dataset and match the values
published in the paper.

> `REPORT.md` contains a "Files in this folder" listing generated before this
> cleanup; that listing is stale. Everything above the listing is current.
