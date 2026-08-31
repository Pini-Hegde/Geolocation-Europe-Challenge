# Proposed deletions — NOT yet executed

| # | what | size | recoverable? |
|---|---|---:|---|
| 1 | `failed_experiments/**/*.pt` (88 old checkpoints, T2–T6 + PoC proxy runs) | ~2.7 G | **yes** — 71 tracked in git (`git checkout -- <path>`); 6 PoC proxy runs are NOT tracked |
| 2 | `outputs/diag/*.npz` (12 diagnostic embedding files) | 558 M | **no**, but regenerable in ~10 min via `scripts_diag_embed*.py` |
| 3 | `submission.zip` (previous 139.8 km submission archive) | 109 M | **yes** — tracked in git |
| 4 | `outputs/poc/off_atto448_f*` (ensemble candidate, 116–128 km, unused) | 180 M | **no** — would need retraining (~35 min/fold) |

Total: ~3.5 G of 4.6 G.

**Kept regardless:** `best_result/`, `src/`, `writeup/`, `submission/`, `outputs/protocol/`,
`failed_experiments/**/*.json` + `history.json` + `config.json` (the writeup needs the
*numbers*, not the weights), all `.md` reports, `slurm/`, `slurm_logs/`.
