# SoftRole research work record

## Scope and instructions

User request, 25 September 2026: extensively review the existing SoftRole-HetGAT ideas, collect primary papers in `papers/`, ground the extension of HetNet/HetGAT in mathematics, compare ROMA, and assess frozen-policy generalization under changing team membership and capabilities.

User clarification during review: keep this work record in `softrole/agents.md`; keep changes within `marl-comm/softrole/`; focus on mathematical foundations rather than implementation. Do not modify application/training code or run a training study as part of this task. Preserve the two original July documents and the supplied `papers/HetNet.pdf`.

## Work completed

- Read the original architecture and critical review with parallel audits of their assumptions, equations, literature and repository references.
- Extracted and reviewed Guide 3, Team Changes and Information-Role Recovery, as the connection to sensing-role handover.
- Inspected current repository read-only to identify feasibility limitations; no implementation code changed. Constructed existing environment variants only to measure observation/action dimensions; no training was run.
- Acquired primary PDFs and text extractions inside `papers/`, with per-area manifests containing source URLs, access status and hashes.
- Verified that ROMA has dynamic roles, RODE evaluates policy transfer to larger teams, CASH studies online capability degradation, and GPL/HOLA cover open teams. Recent hidden-capability and role-communication papers are included, with full-text access limitations flagged.
- Reconstructed original HetGAT class-wise attention normalization; distinguished it from the global-softmax mechanism in the old proposal.
- Prepared mathematical audit with equivariance proof, one-hot reduction conditions, observability barriers, count/size counterexamples and conditional stability bounds.
- Completed the main feasibility report with 38 displayed mathematical blocks, independent proof/literature review, and a readable PDF export.
- Consolidated primary-paper records, BibTeX, file hashes, availability flags and reading order. The final g-MAT check is included: 43 works tracked, 39 main PDFs available, and four full-text access limitations recorded.

## Deliverables and ownership

| File or directory | Purpose | Status |
|---|---|---|
| `SOFTROLE_FEASIBILITY_REPORT.md` | Main synthesis, equations, proofs, feasibility and research design | Complete; primary agent, independently reviewed |
| `research/MATHEMATICAL_AUDIT.md` | Independent mathematical reconstruction and proof details | Complete; math audit agent |
| `research/LITERATURE_ROLE_OPEN_TEAMS.md` | ROMA/RODE/role-communication/open-team evidence audit | Complete; literature agent |
| `research/CAPABILITY_AND_RECOVERY_EVIDENCE.md` | Capability inference, CASH, recovery and latest close work | Complete; primary agent |
| `research/REPOSITORY_FEASIBILITY.md` | Read-only compatibility audit; background, not an implementation | Complete; repository audit agent |
| `papers/manifest_*.json` | Source-specific paper provenance | Complete |
| `papers/README.md`, `papers/manifest.json`, `papers/references.bib` | Consolidated reading map and library | Complete |

## Scientific decisions

- Use agent-permutation **equivariance** for actors and invariance for team summaries; do not infer cross-size return guarantees from symmetry.
- Separate physical capability, inferred capability, computational role, and independently trained partner policy.
- Treat a recurrent frozen policy as adapting through state/belief updates, not parameter updates.
- Distinguish announced failure from hidden failure diagnosis and episode-level variation from within-episode membership changes.
- Present new mathematical propositions as derived conditional properties, not published theorems or experimental findings.
- Prioritize a mathematical model and falsifiable research protocol. Architecture and code remain unimplemented.

## Validation and material corrections

- Independently reviewed the main report’s mathematical statements and prior-art comparisons.
- Corrected the distinction between regular graphs and vertex-transitive/automorphism-symmetric graphs.
- Explicitly stated the joint-history, common-kernel, initial-distribution, and action-factorization assumptions behind the return coupling bound.
- Rechecked RODE’s 50k target-task samples and action-encoder fitting; corrected its transfer description and GIRA’s homogeneous-agent scope.
- Validated local artifact links, PDF signatures, SHA256 records and balanced TeX math delimiters.
- PDF export uses temporary rendering tooling; no repository dependency changes. Dense tables and equations were visually inspected; the mathematical compile had no overflow, missing-character or undefined-command warnings.
- `git -C marl-comm status --short` shows only `?? softrole/`, consistent with all project edits staying in this folder.
- No application code was modified; no training, model implementation, or project test suite was run. Copies of published HetNet code under `research/source_code_audit/` are pinned read-only research evidence, not application modifications.

## Completion

All requested research deliverables are complete. The final report and mathematical/source audits distinguish derived results, published evidence, assumptions and open empirical questions. The paper index records inaccessible full texts explicitly. The original proposal files and supplied HetNet PDF remain preserved.

## Follow-up: implementation-design report (25 September 2026)

User asks for a second report describing how to build and test SoftRole on the **original HetNet codebase**, using its environments and frozen-policy experiments. Current `commstudy` implementation belongs to a different research direction and is not the implementation base for this report. Inspect author code for similar methods and link the design to the feasibility report's mathematical results.

This authorizes an implementation **design report and read-only code research**, not model implementation or training. All new reports, provenance records and unmodified upstream evidence snapshots stay inside `softrole/`. Preserve previously staged files; do not alter the index.

Work in progress:

- Primary agent: original HetNet actor/graph/trainer audit and report synthesis.
- Environment audit: PP, PCP and FireCommander source contracts; benchmark versus event extensions.
- Related-code audit: official ROMA/RODE/CASH/GPL and other close code, pinned references.
- Mathematical protocol audit: theorem-linked checks and frozen evaluation contract.
- New unmodified HetNet snapshot: `research/implementation_sources/hetnet/`, pinned commit `bff9f7f9a9e905d96c6d9762c2c853c9ee2f96ec`.
