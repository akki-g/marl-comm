# Research and implementation references

**Checked for this specification:** September 19, 2026.  
**Repository baseline:** `akki-g/marl-comm`, commit `9eb028ec5ad66bf4dd105623206ffd1c8da7561b`.

The research papers motivate experimental and architectural distinctions. The project files establish what this repository currently implements. The suite's 90-run allocation, seed namespaces, 131,072-frame MAPDN proposal, and workflow design are **new specification choices**, not numerical prescriptions from these papers.

The reference review supports this implementation contract; it is not an exhaustive novelty review or a new numerical reproduction. Original scientific artifacts on the user's local/cluster filesystem remain to be inspected by the implementing agent. Repository links can require the user's existing GitHub access.

## A. Primary research sources

### [P1] BenchMARL — preserve the benchmarking framework

Bettini, M., Prorok, A., and Moens, V. **BenchMARL: Benchmarking Multi-Agent Reinforcement Learning.** *Journal of Machine Learning Research*, 25(217):1–10, 2024.

Primary source: https://www.jmlr.org/papers/v25/23-1612.html

**Relevant contribution:** A systematic MARL benchmarking framework organized around algorithms, models, and environments, with TorchRL as its backend.

**Use in this specification:** Extend the existing model/task/configuration interfaces; do not replace the training loop or create a second PPO implementation. The paper's reproducibility objective motivates explicit resolved configurations, but the exact manifest schema here is a project design.

### [P2] MAPPO — distinguish algorithm from communication module

Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., and Wu, Y. **The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.** *NeurIPS*, Datasets and Benchmarks Track, 2022.

Primary source: https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html  
Author manuscript: https://arxiv.org/abs/2103.01955

**Relevant contribution:** Strong cooperative PPO baselines and analysis of implementation/hyperparameter choices.

**Use:** Keep MAPPO constant and label the experiment a communication-module comparison. The PCP and MAPDN hyperparameters in the specification come from project protocols, not from an assertion that this paper used those exact settings.

### [P3] CommNet — inspiration for Broadcast

Sukhbaatar, S., Szlam, A., and Fergus, R. **Learning Multiagent Communication with Backpropagation.** *NeurIPS*, 2016.

Primary manuscript: https://arxiv.org/abs/1605.07736  
Proceedings: https://papers.nips.cc/paper_files/paper/2016/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html

**Relevant contribution:** Continuous learned communication for cooperative agents.

**Use:** Explain the mean-aggregation lineage of `BroadcastComm`. The project's one-slot feedforward MAPPO implementation is an adaptation; do not label its scores as an exact CommNet reproduction.

### [P4] IC3Net — inspiration for Gated

Singh, A., Jain, T., and Sukhbaatar, S. **Learning When to Communicate at Scale in Multiagent Cooperative and Competitive Tasks.** *ICLR*, 2019; manuscript initially posted in 2018.

Primary manuscript: https://arxiv.org/abs/1812.09755  
Primary conference record: https://openreview.net/forum?id=rye7knCqK7

**Relevant contribution:** Controlled communication using a gate, with the original method's own controller and reward/credit-assignment design.

**Use:** Include the existing gated module, but state that its soft sender gate under common MAPPO does not reproduce all of IC3Net. Do not infer packet savings solely from reduced message amplitude.

### [P5] TarMAC — inspiration for Attention

Das, A., Gervet, T., Romoff, J., Batra, D., Parikh, D., Rabbat, M., and Pineau, J. **TarMAC: Targeted Multi-Agent Communication.** *ICML*, PMLR 97:1538–1546, 2019.

Primary source: https://proceedings.mlr.press/v97/das19a.html

**Relevant contribution:** Learning what to communicate and whom to address; targeted communication and multiple communication rounds.

**Use:** Motivate an Attention comparison separate from Broadcast. Document the project's feedforward, one-round adaptation rather than claiming a faithful recurrent TarMAC baseline. Include transmitted key/value payload in resource reporting.

### [P6] DGN — inspiration for Graph

Jiang, J., Dun, C., Huang, T., and Lu, Z. **Graph Convolutional Reinforcement Learning.** *ICLR*, 2020; manuscript first posted in 2018.

Primary manuscript: https://arxiv.org/abs/1810.09202  
Author publication page: https://z0ngqing.github.io/publication/dgn/  
Conference record: https://openreview.net/forum?id=HkxdQkSYDB

**Relevant contribution:** Graph-based relation representations/message passing and temporal relation regularization.

**Use:** Explain GraphComm's lineage while preserving the distinction between that module and DGN's full learning procedure. The OpenReview page presented an access challenge during this review; the manuscript and author record were accessible.

### [P7] GATv2 — identify the actual graph attention operation

Brody, S., Alon, U., and Yahav, E. **How Attentive are Graph Attention Networks?** *ICLR*, 2022.

Primary manuscript: https://arxiv.org/abs/2105.14491

**Relevant contribution:** Distinguishes static from query-dependent dynamic graph attention and introduces GATv2.

**Use:** The project Graph code describes GATv2-style additive relation scoring. Check the code rather than assuming that Graph with a full mask equals dot-product Attention. This citation does not imply our module is an unchanged external GATv2 implementation.

### [P8] MAPDN — domain semantics and physical endpoints

Wang, J., Xu, W., Gu, Y., Song, W., and Green, T. C. **Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks.** *NeurIPS*, 34:3271–3284, 2021.

Primary source: https://proceedings.neurips.cc/paper/2021/hash/1a6727711b84fd1efbb87fc565199d13-Abstract.html  
Author environment repository: https://github.com/Future-Power-Networks/MAPDN

**Relevant contribution:** MARL active voltage control on distribution networks and an associated evaluation environment.

**Use:** Preserve inverter/voltage-control semantics and separate voltage quality, reactive effort, losses, and solver failures. Upstream supported algorithms/scenarios are not evidence that this project ran them. The integrated research adapter has its own alignment/split/capability rules, documented in [R12–R15].

### [P9] Useful communication is not merely signaling

Lowe, R., Foerster, J., Boureau, Y.-L., Pineau, J., and Dauphin, Y. **On the Pitfalls of Measuring Emergent Communication.** *AAMAS*, 2019.

Primary manuscript: https://arxiv.org/abs/1903.05168  
Author organization record: https://ai.meta.com/research/publications/on-the-pitfalls-of-measuring-emergent-communication/

**Relevant contribution:** Demonstrates that intuitive communication indicators can be misleading.

**Use:** Report actual domain outcomes and frozen-policy interventions separately from message activity, attention visualizations, and action influence. The specific live/severed evaluator is project infrastructure, not claimed as a new contribution of this spec.

### [P10] Statistical evaluation of deep RL

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., and Bellemare, M. G. **Deep Reinforcement Learning at the Edge of the Statistical Precipice.** *NeurIPS*, 2021.

Primary source: https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html  
Primary manuscript: https://arxiv.org/abs/2108.13264  
Author project: https://agarwl.github.io/rliable/

**Relevant contribution:** Highlights uncertainty in few-run RL evaluation and develops stronger reporting tools.

**Use:** Show independent training seeds, interval estimates, and transparent comparisons. Do not treat evaluation episodes as independent learning runs or claim that five seeds are automatically adequate. The per-task paired bootstrap here is a scoped design; do not imply the paper proves its validity for every small-sample inference.

### [P11] Scope information and memory claims

Tessera, K.-a., Hinckeldey, L., Zamboni, R., Abel, D., and Storkey, A. **Probing Dec-POMDP Reasoning in Cooperative MARL.** *AAMAS*, 2026, as recorded in the paper's primary metadata.

Primary manuscript: https://arxiv.org/abs/2602.20804  
Related DOI: https://doi.org/10.65109/ECCJ1033

**Relevant contribution:** Diagnoses memory, teammate-information, and coordination demands across cooperative MARL scenarios.

**Use:** Keep the present feedforward mechanism comparison scoped. It does not establish that local memory cannot substitute for communication or that a task requires the intended reasoning.

### [P12] Capacity versus communication bandwidth

Canesse, A., Goupil, B., Read, J., and Vanier, S. **Decoupling Communication from Policy: Robust MARL under Bandwidth Constraints.** 2026 preprint, arXiv:2605.21085 (SLIM).

Primary manuscript: https://arxiv.org/abs/2605.21085

**Relevant contribution:** Separates policy latent capacity from communication bandwidth and defines a normalized bandwidth budget.

**Use:** Retain active local-capacity controls, distinguish message width from overall actor capacity, and report actual payload. This is methodological context, not a requirement to implement SLIM or a claim that capacity controls are novel.

### [P13] VMAS — the movement simulator

Bettini, M., Kortvelesy, R., Blumenkamp, J., and Prorok, A. **VMAS: A Vectorized Multi-Agent Simulator for Collective Robot Learning.** 2022 manuscript, associated with DARS 2022.

Primary manuscript: https://arxiv.org/abs/2207.03530

**Relevant contribution:** A vectorized PyTorch multi-agent simulation framework and task interface.

**Use:** Preserve the existing VMAS scenario interface and profile actual selected CPU/GPU execution. The published throughput examples do not predict our corrected PCP or MAPDN runtime.

## B. Project evidence and code references

All links below are pinned to the inspected revision. Source contents establish current behavior; narrative completion claims remain attributed to repository records. The agent must recheck the current local revision and access original artifacts for actual result reproduction.

| ID | Source | Why the implementing agent needs it |
|---|---|---|
| R0 | `MARL_Thesis_Strategy_Report.pdf`, supplied September 19, 2026; baseline audit at `ea5bfb4…` | Pages 4–7: evidence limits and source gaps. Pages 15–19: benefit/reliance, evaluation, and implementation cautions. Page 21: MAPDN domain limitations. |
| R1 | `docs/PCP_VISIBILITY_BUDGET_PLAN.md` | D4 task dimensions, visibility, controls, endpoints, seeds, and intervention semantics. |
| R2 | `docs/MAPDN_IMPLEMENTATION_STATUS.md` | September 14 completed comparison and historical Identity-only result. |
| R3 | `docs/MAIN_DIRECTION_NEXT_PHASE_PLAN.md` | Original MAPDN task/optimizer/bank settings and bounded comparison scope. |
| R4 | `docs/RESEARCH_GRADE_PROGRESS.md` | Corrected task/RNG/critic versions and compatibility constraints; historical statuses are dated. |
| R5 | `agents.md` | Latest September 14 completion entry; do not treat an old top-level README snapshot as current evidence. |
| R6 | `docs/communication_methods.md` | Shared interface, implementation differences from papers, masks, and message-cost conventions. |
| R7 | `src/commstudy/communication/graph.py` | Runtime-mask precedence, supported fallbacks, additive relation scoring, and rounds. |
| R8 | `src/commstudy/communication/local_control.py`, `identity.py`, `broadcast.py` | Active local capacity versus pass-through versus exchanged messages. |
| R9 | `docs/RESULTS_v2_simple_spread.md` | Historical method coverage and branch sizes; retain its original overstatements as history, not unquestioned conclusions. |
| R10 | `configs/protocols/pcp_corrected_v1.yaml` | Existing candidate structure and hparams; not automatically authorized as the new suite. |
| R11 | `scripts/evaluate_pcp_budget.py` | Explicit D4-only guard, checkpoint requirements, and audit helpers to factor/reuse. |
| R12 | `src/commstudy/tasks/torchrl/power_grids.py` | MAPDN dimensions/validation, actual critic information, metrics, and runtime contract. |
| R13 | `scripts/run_mapdn_confirmation.py` | CPU runtime, historical `_spec`, bank metrics, finite checks, and physical denominators. |
| R14 | `scripts/run_main_direction_phase.py`; `src/commstudy/experiments/next_phase.py` | Three-method restriction, old seed-locked calibration, phase/bank generation. |
| R15 | `src/commstudy/tasks/torchrl/mapdn_data.py`; `scripts/fetch_mapdn_data.py` | Data/split helpers to inspect in implementation; paths were verified, not a fresh complete audit of every helper. |
| R16 | `scripts/evaluate_mapdn_paired.py` | Paired final checkpoint evaluation, strict reconstruction, full replay, and its real CLI. |
| R17 | `slurm/pcp_03_main_comparison.sbatch` | Historical fixed 30-row GPU array and manifest-bound launch—not the requested new 60-row PCP suite. |
| R18 | `slurm/newton_env.sh` | Historical module/venv defaults, CUDA default, and allocated-CPU-derived numerical thread behavior. |

### Pinned source links

**R0**

- [MARL_Thesis_Strategy_Report.pdf](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/MARL_Thesis_Strategy_Report.pdf)

**R1**

- [docs/PCP_VISIBILITY_BUDGET_PLAN.md](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/docs/PCP_VISIBILITY_BUDGET_PLAN.md)

**R2**

- [docs/MAPDN_IMPLEMENTATION_STATUS.md](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/docs/MAPDN_IMPLEMENTATION_STATUS.md)

**R3**

- [docs/MAIN_DIRECTION_NEXT_PHASE_PLAN.md](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/docs/MAIN_DIRECTION_NEXT_PHASE_PLAN.md)

**R4**

- [docs/RESEARCH_GRADE_PROGRESS.md](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/docs/RESEARCH_GRADE_PROGRESS.md)

**R5**

- [agents.md](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/agents.md)

**R6**

- [docs/communication_methods.md](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/docs/communication_methods.md)

**R7**

- [src/commstudy/communication/graph.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/src/commstudy/communication/graph.py)

**R8**

- [src/commstudy/communication/local_control.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/src/commstudy/communication/local_control.py)
- [src/commstudy/communication/identity.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/src/commstudy/communication/identity.py)
- [src/commstudy/communication/broadcast.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/src/commstudy/communication/broadcast.py)

**R9**

- [docs/RESULTS_v2_simple_spread.md](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/docs/RESULTS_v2_simple_spread.md)

**R10**

- [configs/protocols/pcp_corrected_v1.yaml](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/configs/protocols/pcp_corrected_v1.yaml)

**R11**

- [scripts/evaluate_pcp_budget.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/scripts/evaluate_pcp_budget.py)

**R12**

- [src/commstudy/tasks/torchrl/power_grids.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/src/commstudy/tasks/torchrl/power_grids.py)

**R13**

- [scripts/run_mapdn_confirmation.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/scripts/run_mapdn_confirmation.py)

**R14**

- [scripts/run_main_direction_phase.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/scripts/run_main_direction_phase.py)
- [src/commstudy/experiments/next_phase.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/src/commstudy/experiments/next_phase.py)

**R15**

- [src/commstudy/tasks/torchrl/mapdn_data.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/src/commstudy/tasks/torchrl/mapdn_data.py)
- [scripts/fetch_mapdn_data.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/scripts/fetch_mapdn_data.py)

**R16**

- [scripts/evaluate_mapdn_paired.py](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/scripts/evaluate_mapdn_paired.py)

**R17**

- [slurm/pcp_03_main_comparison.sbatch](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/slurm/pcp_03_main_comparison.sbatch)

**R18**

- [slurm/newton_env.sh](https://github.com/akki-g/marl-comm/blob/9eb028ec5ad66bf4dd105623206ffd1c8da7561b/slurm/newton_env.sh)

## C. Official systems documentation

### [S1] Slurm job arrays

SchedMD. **Job Array Support.** Accessed September 19, 2026.

https://slurm.schedmd.com/job_array.html

Use for array indices, concurrency limits, array environment variables, `%A/%a` output names, and dependency behavior across arrays. Installed Slurm limits must still be checked on the target cluster.

### [S2] Slurm sbatch

SchedMD. **sbatch manual.** Accessed September 19, 2026.

https://slurm.schedmd.com/sbatch.html

Use for literal directive parsing, option placement, dependency meanings, and `--parsable` output including an optional cluster suffix. Workflow safety, manifest locks, and artifact gates are project design requirements layered on those documented APIs.

### [S3] Slurm srun

SchedMD. **srun manual.** Accessed September 19, 2026.

https://slurm.schedmd.com/srun.html

Use for launching job steps and checking behavior on the installed Slurm version. Do not assume current public documentation exactly matches the target version or site policy.

### [S4] UCF ARCC Newton

University of Central Florida Advanced Research Computing Center. **About Newton.** Accessed September 19, 2026.

https://arcc.ist.ucf.edu/index.php/resources/newton/about-newton

The official resource page identifies mixed GPU hardware. It does not establish the user's current account entitlement, module choices, partition permissions, filesystem quota, or job limits. Discover those in the user's authorized environment. An attempted additional generic Slurm-help page was unavailable; no site-policy claim in this spec relies on it.

## D. Claims-to-references checklist for the implementation report

| Report statement | Required backing |
|---|---|
| “All five configurations run on both tasks.” | New real execution evidence; class availability alone is insufficient. |
| “This is a fair fixed-MAPPO mechanism comparison.” | Resolved per-task configuration diff plus [P1–P2] and adaptation descriptions [P3–P7]. |
| “LocalCapacity controls network size.” | Actual active branch counts and local-gradient tests [R8], with scope limits [P12]. |
| “Communication helps.” | Separate-policy native-endpoint effects plus uncertainty; message norms alone do not establish it [P9–P10]. |
| “The trained policy relies on messages.” | Same-checkpoint paired intervention evidence, with input-shift caveat [P9, R1, R16]. |
| “Communication is necessary.” | Not supported by this suite; see [P11] and [R0]. |
| “Graph uses feeder topology.” | Explicit bus-to-agent construction and observed effective mask; a class name is insufficient [R7, R12]. |
| “MAPDN is safe.” | Not established by solver success or average violations; retain physical failure/tradeoff reporting [P8, R12–R13]. |
| “Cluster ready.” | Actual target-runtime preflight/smokes/qualification, not only shell syntax checks [S1–S4]. |
