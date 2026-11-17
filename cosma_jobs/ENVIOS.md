# Submission log

One entry per submission to COSMA (analysis or simulation), newest first. The
per-submission `plan.json` that `submit_missing.py` writes lives under
`analysis_results/` on COSMA, which is not tracked; this file is the record in
the repository. Fill the job IDs with what the submit command printed.

| Field | Meaning |
|---|---|
| Date | Day of submission |
| Jobs | Slurm job IDs (array, review) |
| Code | Commit of the COSMA checkout when submitting (`git rev-parse --short HEAD`) |
| OUT | Output folder |
| Runs | Histories in the plan (`anisotropy_adios2/psc_*`) |
| Why | What the results are for |
| Result | Filled when the jobs end |

---

## 2026-11-17 — firehose 40 d_i (array 12366111, review 12366112)

- **Jobs:** 12366111 (array of 3, one at a time), 12366112 (review, after the array)
- **Code:** b959f6f5d (COSMA checkout pulled 2026-11-16)
- **OUT:** `analysis_results/tesis_firehose_20261117` (`--family firehose`)
- **Runs (all partial histories, last field step / expected):**
  - `psc_firehose_bimaxwellian_strong_bigbox40_12062436` — 1 056 000 / 1 200 000 (88 %)
  - `psc_firehose_bikappa3_bigbox40_11657093` — 919 000 / 1 200 000 (77 %)
  - `psc_firehose_bimaxwellian_moderate_bigbox40_11643619` — 954 000 / 1 200 000 (80 %)
- **Why:** second instability family of the systematic article: bi-Maxwellian vs
  kappa 3 at beta_par = 10, A = 0.1 (the moderate one has no kappa twin). The pair
  is compared up to the shorter history, step 919 000.
- **Left out:** `psc_whistler_bikappa3_strong_12032242` (moved to `staging/`):
  truncated archive, about 1 in 4 field outputs, 576^2 with 1000 ppc, so not the
  numerical twin of the bi-Maxwellian whistler (2304^2, 2000 ppc, mi/me = 800);
  `analysis_config.json` reconstructed by hand.
- **Result:** _pending_

## 2026-11-17 — mirror kappa 3 isotropic control (planned)

- **Jobs:** _to fill_
- **OUT:** `analysis_results/tesis_mirror_k3control_20261117`, with
  `RUN_ROOT=staging/k3control` (a link to the one run)
- **Runs:** `psc_mirror_bikappa3_isotropic_12075009`
- **Why:** missing numerical control of the kappa 3 mirror run (tail relaxation,
  background nu_0, scaled control of the energy budget).
- **Result:** _pending_

## 2026-11-13 — mirror moderate series, controls and one whistler (array 12350652, review 12350653)

- **Jobs:** 12350652 (array), 12350653 (review)
- **OUT:** `analysis_results/tesis_pendientes`
- **Runs:** `psc_mirror_bimaxwellian_moderate_11596993`, `psc_mirror_bikappa5_moderate_12063822`,
  `psc_mirror_bikappa3_moderate_11618877`, `psc_mirror_bimaxwellian_isotropic_12075007`,
  `psc_mirror_bikappa5_isotropic_12075008`, `psc_whistler_bimaxwellian_strong_mr800_12068623`
- **Why:** current pipeline on the mirror series with its isotropic controls.
- **Result:** analysed and downloaded to `analysis_results/analisis_cosma/tesis_pendientes`.
  The plan of that version left out short production histories (firehose) and the
  kappa 3 isotropic control.
