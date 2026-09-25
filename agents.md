# Working on commstudy

The active workflow is two configs, five communication modules, two thin Python
entrypoints, and one Slurm wrapper. Keep the implementation in `src/commstudy/`.
Package code must never import scripts or depend on repository-relative YAML.

- Preserve task returns, observation/critic separation, masks, PPO replay checks,
  finite-value checks and evaluation RNG isolation.
- Each config expands to all five methods × one to five distinct seeds.
- Keep worker outputs separate; the coordinator owns combined tables.
- Preserve failed attempts and historical results. Do not interpret engineering
  smoke tests as evidence of a communication benefit.
- Test changed behavior with `uv run --locked pytest`; lint with
  `uv run --locked ruff check src scripts tests`.
- Document the current workflow in README. Do not append operational diaries or
  add study-specific supervisors, approval chains, or one-off scripts.

Historical instructions and research records are indexed in
[docs/HISTORY.md](docs/HISTORY.md). The current user-authorized simplification
supersedes the old source-freeze and phase-specific workflow instructions.
