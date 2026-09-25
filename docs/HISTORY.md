# Historical workflows

The complete pre-cleanup tree is preserved by the annotated Git tag
`pre-streamline-2026-09-24`, at commit
`21d79cae390c03bd0f507626b3b3d93f304177e3`.

To inspect historical code without changing the current checkout:

```bash
git show pre-streamline-2026-09-24:agents.md
git worktree add --detach ../commstudy-history pre-streamline-2026-09-24
```

That revision contains the original `scripts/`, composed `configs/`, Slurm
pipelines, `src/commstudy/experiments/` protocols and supervisors,
`vendor/mapdn-source/`, and their tests. It also contains the historical
operational diary and study plans under `docs/`.

| Historical topic | Paths at the tag |
| --- | --- |
| Initial communication comparison | `docs/RESULTS_v2_simple_spread.md`, `docs/RESULTS_pcp_pilot.md` |
| PCP qualification and confirmation | `docs/PCP_PROTOCOL.md`, `docs/PCP_CONFIRMATION_*` |
| PCP visibility/budget studies | `docs/PCP_VISIBILITY_BUDGET_*`, `scripts/run_pcp_budget_*` |
| MAPDN integration and confirmations | `docs/MAPDN_*`, `vendor/mapdn-source/` |
| Later paired/intervention studies | `docs/MAIN_DIRECTION_NEXT_PHASE_PLAN.md`, `src/commstudy/experiments/next_phase.py` |
| Mechanism suite and its evidence | `docs/experiments/pcp_mapdn_mechanisms_v1/`, `configs/protocols/*mechanisms*` |

Local `runs/`, `results/`, generated BenchMARL directories, scientific result
documents, notebook, and all 526 scheduler `.out` files are preserved. Generated
files are ignored instead of tracked in the active tree. Sibling projects were
not modified. Historical output should be interpreted with its original source
and protocol; the new runner deliberately does not emulate the old CLIs.

MAPDN simulator source originated from Jianhong Wang and contributors' MAPDN
implementation. The existing source-origin record and MIT license were moved
unchanged into `src/commstudy/environments/mapdn/`. Required physics and corrected
adapters were retained; unused agents, algorithms, trainers and rendering were
retired. The original vendor tree, including its source patch, remains at the tag.
