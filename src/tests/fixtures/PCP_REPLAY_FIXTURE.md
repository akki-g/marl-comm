# PCP replay roundoff regression fixture

`pcp_replay_roundoff_seed010.pt` contains the real CPUv1 seed10 Identity actor
at480,000 frames, its saved6000 observations, and constructed valid actions
with standard-normal support point z=(-3,-1). It is a numerical regression
fixture, not the unavailable564k failure batch or scientific confirmation.

The source checkpoint SHA256 is
`eaa7caed42a5fd41a67617c43b79fab8471bd2d801ea1be6c93506357c415a7f`.
On the original macOS/ARM PyTorch2.13.0 single-thread runtime, an unchanged
actor evaluated as[10,600] produces two log-probability tolerance violations;
replay at collection width10 is bitwise exact. The test requires unchanged
rtol=atol=1e-5 and also verifies that changed weights still fail. Other CPU
kernels may produce different floating-point discrepancies.

Full probe scripts, hashes, and separate sampled-action and constructed-point
results are retained under
`results/pcp_candidate_confirmation_cpu_v1/replay_diagnostic/` locally. This
small fixture is included with the tests so they do not require ignored runs.
