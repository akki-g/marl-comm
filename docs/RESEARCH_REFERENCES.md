# Research references

These primary sources explain the algorithms, communication designs and task
semantics. They do not establish a communication benefit in this codebase.
The current execution workflow is in [README](../README.md); prior study
specifications and their source links are retained at the [historical tag](HISTORY.md).

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
