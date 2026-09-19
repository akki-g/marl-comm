# Read-only MAPDN confirmation analysis

`scripts/summarize_mapdn_confirmation.py` produces descriptive analysis from
preserved confirmation JSON, per-transition JSONL, and tidy metrics CSV files.
It uses only the Python standard library. It does not import MAPDN, Torch,
commstudy, or any environment/training code. It does not reload actors or run
simulators. The recorded protocol verdict is copied without recomputation.

Run after the confirmation's `report.json` has been written:

```bash
python3 scripts/summarize_mapdn_confirmation.py \
  --input-dir results/mapdn_case33_bounded_confirmation_v1_20260909 \
  --out-file results/mapdn_case33_bounded_confirmation_analysis_20260909.json
```

The analysis destination must be a new file outside the evidence directory.
Every read input and retained archive is SHA256-bound in the new report, along
with the analysis script. Inputs are rehashed after analysis; changes during a
read abort creation. Frozen-source hashes are reproduced from the preparation
record, with their provenance identified. Archive bytes are independently
rehash-checked against the original receipts; the original archive member
read-back results remain attributed to those receipts.

Both validation and test banks receive pooled and per-episode summaries, with
initial/final pairs matched by episode ID, absolute reset row, and environment
seed. Mismatched or duplicated identities are reported. Each paired effect is
final minus initial, with episode lengths and physical-observation denominators
retained. Missing or failed banks remain visible and cannot change the protocol
verdict.

Diagnostics count team measurements once per environment transition. Failure,
feasibility, reward and curtailment summaries include all attempted transitions.
Voltage, line loss and reactive effort summaries require both `valid=1` and
`voltage_metrics_valid=1`. Their eligible, finite, missing and nonfinite counts
are explicit; missing values produce null complete means. Finite-only means are
labeled separately. Exact bus counts provide voltage-frequency denominators.
Power-sample sums are not energy integrals.

The actor receives zone-local observations: each inverter receives features
for its zone, rather than inverter-only measurements. Voltage is measured in
per-unit values; line-loss and PV power in MW; reactive power in MVAr. Actions
are normalized dimensionless fractions bounded by `[-0.8,0.8]`. Physical-bound
saturation is `abs(action)/0.8 > 0.999`, with explicit action-scalar coverage and
denominators. The inherited generic `abs(action) > 0.999` training metric is not
used for this physical saturation measure.

The source row interval is 180 seconds. The original dataset's timestamp labels
span `2012-01-01T00:03` through `2015-01-01T00:00`; the manifest stores normalized
UTC labels. Those labels do not establish a geographic timezone.

Training summaries retain transition counts and row/diagnostic coverage,
managed status and provenance, parameter/checkpoint checks, optimizer-health
series, and missing/nonfinite observations. Required series include finite
fractions, replay tolerance, losses, gradient norms, approximate KL, entropy,
effective sample size and clipping. Missing frames and duplicate frame entries
are recorded against each seed's frozen collection schedule. These are
descriptive checks and do not introduce a new scientific decision rule.
