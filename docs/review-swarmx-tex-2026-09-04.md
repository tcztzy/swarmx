# Paper Review: SwarmX (SoftwareX submission, reviewed revision `8cbbdbf`)

Reviewer: deep technical pass against `swarmx.tex`, repository artifacts, and external
references. Review date: 2026-09-04. Predecessor rounds: `artifacts/swarmx-softwarex-review`
(R1), `artifacts/swarmx-softwarex-review-r2` (R2), `artifacts/swarmx-paper-review` (earlier draft).

This is a historical review, not a validation report for the current implementation. Component
names and reproduction paths have been normalized to Memory; findings and quoted passages refer
to the reviewed manuscript revision.

## 1. Form compliance (SoftwareX 2026)

- Abstract: 111 words, well inside the 150-word Elsevier ceiling.
- All five required sections present, in order: Motivation and significance, Software
  description, Illustrative examples, Impact, Conclusions.
- Six `\sep`-separated keywords; C8 uses the literal "Support email for questions" label.
- Three figures, five tables, 22 distinct citation keys, all bib entries resolve and
  carry a year/author/title/venue/identifier triple.
- `swarmx.log`: 0 Overfull, 18 Underfull, 0 LaTeX Warning — typesetting is clean.

Verdict: improved over R2; the structural blockers that the R2 review listed are now met.

## 2. Internal claims that were independently verified

| Claim (location) | Verification |
| --- | --- |
| 10.0 / 8.8 days, "12% lower" (Fig. 1, §4.4.1) | Visible in `docs/figures/swarmx-desktop-trace-and-provenance.jpg` (Control 10.0, Treatment 8.8, "12% lower" annotation). |
| "Journal #17" (Fig. 2 caption) | Right-pane "Provenance: Journal #17" present in same figure. |
| SHA-256 digest + Run ID `39b961b9-f304-4fc4-a1a2-cea21a7a18c5` | Both visible in Side View. |
| Three artifact cards (Fig. 1) | CSV / Python script / PNG cards present. |
| 12 tool calls, 52 s wall time (Fig. 1) | Footer "12 tool calls" and "Run for 52 s" both visible. |
| RO-Crate 1.3 export, content addressing | `packages/science/core/src/contracts.ts` exports `RO_CRATE_FORMAT = "ro-crate@1.3"`; research-object test asserts the same. |
| "Open Knowledge Format 0.2" | `docs/memory.md` line 4 links the spec; vault advertises `okf_version: "0.2"`. |
| "DSH packages 0.1.2-rc.1" | `apps/desktop/package.json` pins all five `@deepseek-ai/dsh-*` packages to `0.1.2-rc.1`. |
| Swarm DAG acyclicity | `packages/core/swarm/src/team-policy.ts` line 33 throws on cycles ("Team plan task graph must be acyclic"). |
| Electron renderer hardening | `apps/desktop/src/window.ts` sets `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`; `will-navigate`, `setWindowOpenHandler`, and `setPermissionRequestHandler` are guarded. |
| 1030 guarded / 10701 unguarded states and all per-invariant numbers | Match `examples/softwarex/bounded-model/results/depth-7.json` exactly. |
| Four tasks at team revision 44, 10 SHA-256 lines, four artifacts | `examples/softwarex/swarm-memory/run/manifest.json` and `MANIFEST.sha256`. |
| 596 = 601; 596 + 9 = 605 = 601 + 3 + 1 (test arithmetic) | Holds; `codex-real.test.ts` has two `describe.skipIf` blocks (5 + 3 = 8) plus one skip elsewhere → 9. |
| `pnpm paper:demo` / `pnpm paper:model` exist | `package.json` wires them to `node examples/softwarex/swarm-memory/scripts/verify.mjs` and `tsx scripts/run-swarm-model.ts 7`. |
| DVC 3.67.1, macOS 26.6.2 arm64, pnpm 11.7.0 | `sw_vers` matches; `dvc --version` matches; `package.json` declares `pnpm@11.7.0`. |
| `Memory … Markdown compatible with OKF 0.2`, "no native conversation identifiers in the manifest" | `manifest.json.privacy` lists exactly the four excluded categories. |
| All 2026 external references resolve | `arxiv.org/abs/2605.28282` (Xia/Wang), `2608.18398` (Kim/Miao/Liu), `2608.18312` (Yin/Du/Prince/Cherukara), `10.1016/j.softx.2026.102782` (Cheikh Tourad/Lachgar), `10.1145/3772318.3791101` (Martin-Boyle et al., CHI 2026), `10.1038/s41586-026-10265-5` (Lu et al., Nature 651(8107):914–919) all return the expected titles. |

This is a much higher level of traceability than the average software submission.

## 3. Release blockers (must fix before submission)

1. **C2 is a moving root URL, not an immutable version link.** Code metadata C2 cites
   `https://github.com/tcztzy/swarmx`; SoftwareX requires "the permanent link to the
   code/repository used for this code version". R2 raised this and it was not changed.
2. **The evaluated commit is not on the public remote.** `git rev-parse origin/main` returns
   `c09f70eb118f2e8d274451f8e4cb6f3cc06f5876` while `git rev-parse HEAD` returns
   `8cbbdbf6b46b0637c2424e571a68c58a2a1ffabb`; the latter is unreachable from any remote-tracking
   branch. The fixture manifest itself labels `sourceState: "working-tree evaluation candidate"`.
3. **No `v0.1.0` tag exists.** `git tag -l` jumps from `v0.2.0` to `v0.4.0` and beyond, yet
   every `@swarmx/*` package declares `"version": "0.1.0"`. Without a tag, C2 cannot be made
   immutable.
4. **Licence filename mismatch.** The repository root contains `LICENSE` (not `Licence.txt`).
   The 2026 SoftwareX template still uses `Licence.txt`; this is a desk-screening fix.

### 3a. Tag-history trap (new, must be resolved before any tag or C2 change)

The repository hosts two completely unrelated histories with zero common ancestors:

- `v0.2.0` (2024-11-02) → `v0.7.0` (2025-10-21): a Python project — `pyproject.toml`,
  `src/`, `tests/`, `uv.lock`, `mkdocs.yml`, `mkdocs` dependency. Description
  ("A lightweight, stateless multi-agent orchestration framework") matches the OpenAI Swarm
  lineage that `LICENSE` already credits.
- `v3.1.0` (2025-07-16) → `v3.2.0` (2026-07-26): an Electron + Node monorepo
  (`swarmx-monorepo` `version: "3.2.0"`) with `evals/`, `DESIGNS.md`, `design-qa.md`,
  `mkdocs.yml`.
- `HEAD` `8cbbdbf` and `origin/main` `c09f70e`: another Electron + Node monorepo with
  `apps/`, `native/`, `patches/`, `scripts/`, `CODEBASE.md`, and **no** `DESIGNS.md` /
  `design-qa.md` / `evals/`.

`git merge-base HEAD v3.2.0` is empty; `git merge-base HEAD v0.2.0` is empty. HEAD and
the v3.2.0 line are different code bases with the same name.

All 18 v0.x / v3.x tags are pushed to `origin`; there are also `codex/release-3.1.5` and
`codex/release-3.2.0` branches on origin that point at the v3.x line.

**Implication.** A naïve `git tag v0.1.0 && git push origin v0.1.0` on HEAD would create a
repository with `v0.1.0` and `v3.2.0` that are not comparable — a SoftwareX reviewer
opening the GitHub tags page would see a version regression with no commit connecting
them. Three safe options:

A. **Push only, commit-pinned C2, no v0.1.0 tag.** Run `git push origin 8cbbdbf:main`
   (fast-forward, low risk) and change C2 to
   `https://github.com/tcztzy/swarmx/tree/8cbbdbf6b46b0637c2424e571a68c58a2a1ffabb`.
   The version is still 0.1.0 in the package manifests; the immutable version identity is
   the commit SHA. Add one sentence in C7 ("the v0.2.0–v3.2.0 tags predate the SwarmX
   fork and refer to an unrelated code base").
B. **Push + delete the orphan tags.** Same as A, plus `git push origin --delete <tag>`
   for each of the 18 v0.x / v3.x tags. This is destructive and irreversible from origin
   (recoverable only by a fresh clone with reflog). Should only be done if the authors
   control the upstream project as well.
C. **Push + add a non-conflicting label.** `git tag swarmx-softwarex-0.1.0` or
   `git tag softwarex-snapshot-2026-09-04` on HEAD. Leaves the orphan history visible but
   does not pretend to compete with it.

I will not execute any of (A), (B), (C) without explicit confirmation.

## 4. Content weaknesses (in priority order)

5. **Impact is grounded in author-run synthetic data only.** §5 maps Mokyr's knowledge costs
   onto SwarmX functions entirely by argument; both illustrative examples use fixed synthetic
   inputs (recovery time, germination). No third-party user, no disciplinary adoption, no
   non-SwarmX baseline. SoftwareX's "good–substantial contribution" bar typically requires
   external evidence; otherwise the section reads as potential impact, not demonstrated impact.
6. **Boundary with the closest concurrent work is thin.** ResearchLoop, LEDGER, PaperTrail and
   the Yin et al. observability profile all overlap SwarmX's claim–evidence layer. Table 2 has
   one-line "Connection to SwarmX" entries; a feature/authority/evidence matrix (or a short
   paragraph) is needed to make the contribution boundary clear. Currently a reader cannot tell
   whether SwarmX re-implements a ResearchLoop-style claim ledger or strictly links to it.
7. **Mokyr / 2025 Nobel framing is borderline.** The paper says the useful-knowledge framework
   is "central to the scientific background for the 2025 Prize in Economic Sciences" and then
   presents a four-layer institutional lens "as a synthesis used in this article, not a
   taxonomy proposed by Mokyr." The disclosure is there, but the proximity invites the reader
   to over-read the synthesis as the committee's view. Consider a softer attribution
   ("Mokyr's distinction is one of the influences; the four-layer mapping is this article's
   synthesis").
8. **Codex version is inconsistent across the manuscript.** C6 and §4.5.2 reference
   `Codex CLI 0.153.0-alpha.5`; the fixture manifest records `codexCliVersion: "0.150.1"`.
   If 0.153 is the 4 September rerun and 0.150 is the 2 September fixture, that should be said.
9. **Default-run test failure is not visible in Code availability.** §4.5.2 reports the
   unmasked default run failed one Codex hydration test and is treated as a "live-adapter
   compatibility limitation" — good honesty. C6 and Code availability do not echo this, so a
   reader skimming metadata may assume the default test command is green. State the boundary
   in both places.
10. **The "12%" is shared between two different CSVs.** The desktop screenshot shows
    `Control 10.0 / Treatment 8.8`; `runScienceDemo` writes `baseline,10\ntreatment,8.8`.
    The number is consistent, but the group labels diverge between the desktop session and
    the reproducible fixture. Make this explicit in §4.4.1 and the caption.
11. **Unguarded model numbers are not a baseline.** §4.5.3 already notes the unguarded run is
    "a fault-injection sanity check, not an empirical baseline". Given that nothing in the
    paper can be compared against the 9,417 / 10,701 = 88% figure, consider demoting it to an
    internal-validation subsection rather than keeping it in the main results table.
12. **DSH release is moving beneath the paper.** C6 cites `DSH packages 0.1.2-rc.1`, but the
    most recent commit on `main` is "Upgrade DSH to 0.1.2-alpha.4". When `v0.1.0` is tagged
    the DSH version must be frozen alongside.
13. **"exactly nine" is precise but opaque.** `codex-real.test.ts` has two `describe.skipIf`
    blocks with 5 and 3 `it()` calls respectively; the ninth comes from a file the paper does
    not name. Either enumerate the nine (cheap and reviewer-friendly) or relax "exactly" to
    "nine, of which eight are in codex-real.test.ts".

## 5. Additional finding (not in R2): OpenAI Swarm lineage

`LICENSE` carries two copyright lines: `Copyright (c) 2024 OpenAI` and
`Copyright (c) 2024-2026 Tang Ziya`. The paper never mentions this lineage, and
`SPEC.md` / `CODEBASE.md` only reference OpenAI in the Codex / Responses-API context, not as a
code ancestor. R2 already flagged that the manuscript does not compare directly with the
OpenAI Swarm orchestration framework; that comparison is still absent, and the dual copyright
makes the omission material. Either acknowledge the upstream in a single sentence (something
like "SwarmX extends the OpenAI Swarm (2024) multi-agent pattern with …") and add a one-row
comparison entry, or replace the dual copyright with a single line and document the fork point
in `CODEBASE.md`. This is a content/accreditation question the authors should decide.

## 6. Smaller items

- §4.4.1 and the abstract should add the word "descriptive" next to "12% lower" so it matches
  the disclaimer visible in the screenshots.
- §4.4.2 mentions "12 tool calls" in Fig. 1 but does not mention the "Run for 52 s" line that
  is also visible; one sentence would tie execution effort to the coordination graph.
- The cite block in §3 (`\cite{brinckman2019,pizzi2016,pimentel2017,samuel2018}`) is correct
  under `biboptions{sort&compress}`; no change needed.
- The "12 tool calls" footer on Fig. 1 is consistent with the `Run for 52 s` figure; the
  Run record ID `39b961b9-…` is identical across Fig. 1 and Fig. 2 — good cross-figure
  consistency.

## 7. Items I could not verify in this sandbox

- I could not run `pnpm test` end-to-end here (sandbox `node-safe-delete-shim` blocks some
  filesystem paths; `pnpm build:lib` would also need to be run first). The 596/9 and 601/3/1
  numbers are internally consistent and match `codex-real.test.ts` structure, but I did not
  independently observe the totals. Recommend attaching `output/test-summary-2026-09-04.txt`
  to the submission.
- I could not run `pnpm paper:demo` or `pnpm paper:model` to completion here. The verification
  code in `examples/softwarex/swarm-memory/scripts/verify.mjs` is a single
  `assert`-driven Node script and the JSON report is fixed by a regression test, so the
  confidence is high — but again, attaching a fresh run log would remove reviewer doubt.

## 8. Scoring (R2 → R3 delta)

- Form: 3.5 → **4.0** (five-section structure, abstract length, keywords, C8 all done).
- Architecture / software description: **4.0** (clear ownership story, Electron hardening
  real, Swarm DAG enforced).
- Evaluation and reproducibility: 3.0 → **3.5** (deterministic fixtures, bounded-model
  numbers, claim-by-claim verification much stronger than R2).
- Impact evidence: 2.0 → **2.5** (synthetic examples in place but no third-party user).
- Overall recommendation: **Major revision**, but the desk-screen steps (C2 immutable link +
  push commit + tag v0.1.0 + Licence.txt) are the only true blockers. The remaining items
  are paper-shaping improvements.

## 9. Concrete actions the authors can take in order

1. Push `8cbbdbf` to `origin`, tag `v0.1.0`, change C2 to a commit-pinned or
   Zenodo-DOI-pinned link.
2. Copy `LICENSE` to `Licence.txt` (or add a `Licence.txt` alias).
3. Align Codex version numbers in C6, §4.5.2, and the fixture manifest; freeze DSH at
   `0.1.2-rc.1` in the v0.1.0 tag.
4. Add a one-paragraph feature/authority/evidence comparison with ResearchLoop, LEDGER,
   PaperTrail and Yin et al. 2026.
5. Move the unguarded model numbers to an "internal validation" subsection and reframe them
   as fault-injection, not a baseline.
6. Re-emphasize "descriptive" alongside "12%" in §4.4.1 and the abstract; add one sentence
   noting the desktop session and `runScienceDemo` differ only in group labels.
7. Commit `output/paper-demo-latest.log` and `output/paper-model-latest.json` so reviewers
   do not need to re-run.
8. Soften the Mokyr / Nobel sentence to "Mokyr's distinction informs this synthesis; the
   four-layer mapping is this article's organizational device".

## 10. Paper-level edits applied in this review pass

The following reviewer-suggested clarifications were applied directly to `swarmx.tex` (and
recompiled clean: 0 LaTeX Warning, 0 Overfull, 18 Underfull — same as the pre-edit baseline).
Remote operations (push, tag, C2 immutable link) are NOT applied and require the authors'
explicit confirmation.

- §1, Mokyr framing: rephrased "central to the scientific background for the 2025 Prize in
  Economic Sciences" to "one of the influences drawn on in the scientific background for the
  2025 Prize in Economic Sciences", and moved the knowledge-cost clause into the committee
  citation so the synthesis is clearly the article's.
- §1, Table 2 introduction: added a four-sentence paragraph explaining SwarmX's relationship
  to ResearchLoop, LEDGER, PaperTrail, and the Yin et al.\ observability profile — namely that
  SwarmX does not maintain its own claim ledger and that the claim-to-evidence path is
  portable through RO-Crate 1.3.
- §4.4.2, fixture provenance: appended one sentence stating the Swarm-to-Memory
  fixture was produced with Codex CLI `0.150.1` (as recorded in `run/manifest.json`) and that
  the 4 September 2026 test rerun used Codex CLI `0.153.0-alpha.5`.
- §4.5.1, skipped tests: replaced "exactly nine native Codex App Server acceptance tests"
  with the precise file/test split: eight from `apps/desktop/tests/codex-real.test.ts` (five
  App Server lifecycle + three full acceptance) plus one peer-registration case in
  `apps/desktop/tests/runtime-platform.test.ts`. Long paths now use `\path{}` so they break
  cleanly inside the line.
- §5, Impact: prepended a single sentence: "Version 0.1.0 has no third-party user at
  submission, so the impact discussion below describes enabled research questions rather
  than demonstrated adoption."
- Code availability: added a sentence noting the unmasked default `pnpm test` run on macOS
  arm64 skips three external Codex cases and fails one Codex hydration case against Codex CLI
  `0.153.0-alpha.5`; the portable configuration and both publication fixtures remain green.
- New file `Licence.txt` at the repository root (copy of `LICENSE`) to satisfy the
  SoftwareX 2026 template filename requirement.

### Still requires human action

- Push `8cbbdbf` to `origin` and tag `v0.1.0` (no `v0.1.0` tag currently exists in the
  repository).
- Replace C2 in Code metadata with a permanent link (commit-pinned URL, tag URL, or Zenodo
  DOI). I did not edit C2 because the right URL depends on the chosen push/tag strategy.
- Decide how to acknowledge the OpenAI Swarm lineage: one sentence in §1 plus a comparison
  row, or a single-author copyright in `LICENSE` with a fork note in `CODEBASE.md`.
