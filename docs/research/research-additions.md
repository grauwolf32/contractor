# Review & additions to research.html (2026-07-07)

Reviewer pass over `docs/research/reports/research.html` (39+4 directions A–AQ, planner variants V0–V10).
Grounded in a read of `tests/eval/scoring.py`, `tests/eval/trace_vuln_scoring.py`,
`deploy/llama.cpp/serve.sh`, `contractor/utils/settings.py`, `scripts/probe_variance.py`,
and the fixture set under `tests/eval/fixtures/`.

## Verdict

The memo is unusually complete and disciplined: falsifiable hypotheses, defined metrics,
mission-asymmetry as the bar, held-out splits (F3), a deterministic-baseline floor (AO), and
GT adjudication (AP). The pipeline lifecycle and the two meta-principles hold the 200-ish
individual A/Bs together coherently.

The gap is **not** in the capability directions — it's that the memo asks ~200 A/B questions of
an eval whose **own** validity is under-instrumented. F/AO/AP fix three validity axes (cost,
deterministic floor, GT completeness). Four more remain unmeasured and sit *upstream* of every
capability conclusion: (1) serving-layer variance, (2) scorer-matcher strictness, (3) fixture
contamination/memorization, and (4) the objective function itself (tokens ≠ the adopter's cost).
A false discovery in the harness invalidates the direction it "confirms," so these are the
highest-leverage additions. They're also the cheapest — all run on the existing harness.

Two structural notes on the existing tests, applied broadly below:
- ~half the mission-asymmetry hypotheses (L3, O3, X4, AE2, AH5…) state the pass-criterion as
  `uplift(27b) > uplift(70b)`, which is **mechanically biased by ceiling effects** — a saturated
  70b baseline can't uplift regardless of mechanism. This silently inflates the mission signal.
- Every A/B rides pass@1–5 on tiny fixtures; the memo already worries about noise (F1, Z2). The
  cheapest variance kill is a paired/common-random-numbers design + a measured serving-noise floor,
  neither of which is in place today (no `--seed`, `-np>1` continuous batching in `serve.sh`).

---

# Part 1 — New hypotheses (house format)

## AR — Eval-validity floor (variance · scorer · multiplicity)
Impact: High (enabler) · Effort: Low · Measurement · protects every A/B

**Thesis.** F/AO/AP hardened cost, deterministic floor, and GT. Three validity axes remain
unmeasured and sit beneath every conclusion in the memo. Instrument them once and every prior
and future A/B becomes trustworthy; skip them and an unknown fraction of the 200 hypotheses are
ranking noise.

**Grounding.**
- `deploy/llama.cpp/serve.sh` runs `-np` parallel slots + continuous batching with **no pinned
  `--seed`**; `model_temperature` defaults to `None` (`settings.py`). llama.cpp numerics depend on
  batch composition, so identical prompts at temp 0 are **not** bitwise-reproducible. AC3/Z1/Z2
  claim "byte-identical injected block ⇒ variance is model noise" — but there is a serving-noise
  term they never isolate. `scripts/probe_variance.py` samples spread but folds it into the A/B
  delta rather than reporting it as a floor.
- `scoring.py` matches on `file(normalised) ∧ cwe∈acceptable_cwes ∧ line∈±LINE_TOLERANCE(=10)`.
  `LINE_TOLERANCE` is a hardcoded module constant, never swept. All 25 fixtures carry
  `acceptable_cwes`/`acceptable_locations`, but that tolerance-set's *completeness* is unaudited —
  the matcher analogue of AP's GT-completeness concern.
- ~200 hypotheses × pass<0.05 ⇒ ~10 false positives expected by chance; the memo has no
  multiplicity discipline (pre-registration / FDR).

**AR1 — Serving-noise floor.** Repeat one fixed config (same prompt, temp 0) N≥10× on the actual
LM Studio/llama.cpp stack; the σ of F1/capture/tokens is the *irreducible* band every effect must
clear — distinct from GT (AP) and retrieval (Z2) variance.
- *Test:* N-repeat sweep at `-np=1` vs `-np>1` (isolates the batching-nondeterminism term).
- *Pass/kill:* Publish `serving_sigma` into the eval/v1 envelope as the promotion threshold. If
  `-np>1` σ ≫ `-np=1` σ, eval runs should pin `-np=1` (trading throughput for attributability).
- *Metric:* σ(F1/capture/tokens) across repeats × parallelism; longest-common-prefix stability (ties A1).

**AR2 — Scorer-strictness sensitivity.** Sweep the matcher tolerances (`LINE_TOLERANCE ∈ {0,10,25,∞}`,
exact-CWE vs CWE-family, exact-file vs file∨caller) and measure how much each headline conclusion
depends on arbitrary thresholds. A real find mislabeled FP by a sibling-CWE or an off-by-15 sink line
is a **matcher artifact**, biasing precision *downward* independently of GT incompleteness (AP).
- *Test:* Re-score existing envelope history under the tolerance grid; also audit `acceptable_cwes`/
  `acceptable_locations` coverage the way AP audits GT.
- *Pass/kill:* Report each conclusion's stability interval; a ranking that flips within the plausible
  tolerance range is scorer-driven, not capability-driven. Freeze one tolerance as canonical + report
  the band.
- *Metric:* rank stability vs tolerance; per-fixture P/R sensitivity; tolerance-set coverage rate.

**AR3 — Multiplicity discipline.** Adopt pre-registration (N + threshold before the run, already
gestured at in the planner-variants "Method") + Benjamini-Hochberg FDR across each batch of A/Bs.
- *Test:* Apply FDR to the already-run A/B corpus; count how many "wins" survive q=0.1.
- *Pass/kill:* The surviving set is the trustworthy roadmap; the culled set returns to "needs pass@N."
- *Metric:* # surviving discoveries at FDR q=0.1 vs raw p<0.05.

## AS — Contamination & generalization to private code
Impact: High (existential eval-validity) · Effort: Low–Med · the mission's true target

**Thesis.** The mission is "any team hardens **their own** product" — i.e. generalization to
*private, unseen* code. But the fixtures are overwhelmingly public: `crapi-*`, `cvebench-cve-*`
(public CVEs with published fixes *and* writeups), `realvuln-vampi/dsvw/dvpwa/pythonssti`,
`spring`, `fastapi` — all heavily represented in pretraining and in named, step-by-step public
walkthroughs. A 27b/70b may **recall** "crAPI's `/community/api/v2/...` has a BOLA" from writeups
rather than derive it from the code. That inflates recall in exactly the way that will **not**
transfer to an adopter's private repo — the one number the whole roadmap exists to move. This is
distinct from prompt-overfit (B4/F3, which the memo covers): here the *model weights* are
contaminated, so no prompt change fixes it and every recall number has an unknown upward bias.

**Grounding.**
- Fixture list (above) is ~90% public teaching apps / disclosed CVEs.
- No perturbation/rename/mutation harness exists (`grep` for perturb/obfuscate/canary → none).
- The memo's anti-overfit machinery guards prompts, never the model's memorization of the fixture.

**AS1 — Perturbation-sensitivity (semantic-preserving).** Mechanically rename endpoints/params/
vars/files and relocate handlers within a fixture (behavior-preserving), then re-score. A model
*deriving* vulns is ~invariant; a model *recalling* them collapses.
- *Test:* Original vs perturbed crapi/vampi/cvebench; Δrecall by fixture.
- *Pass/kill:* Small Δ ⇒ derivation (trustworthy); large Δ ⇒ memorization (that fixture's recall is
  inflated, discount it and re-weight the aggregate). Either way the perturbed set becomes the
  honest benchmark.
- *Metric:* recall(original) − recall(perturbed) per fixture × model size.

**AS2 — Public-vs-fresh gap.** Author (or via P self-play) a small set of *never-published* fixtures
in the same frameworks/classes; compare recall to public fixtures of matched difficulty.
- *Test:* public vs fresh, matched on class×framework×LoC.
- *Pass/kill:* A large public≫fresh gap quantifies the contamination tax and re-baselines the whole
  memo's recall figures toward what an adopter will actually see.
- *Metric:* recall(public) − recall(fresh), matched.

**AS3 — Larger models are more contaminated, not more capable (the mission trap).** Part of the
measured 27b↔70b gap may be *memorization*, not *reasoning* — in which case "close the gap" is
partly chasing the 70b's recall of writeups the adopter's private code won't match.
- *Test:* AS1's perturbation Δ, stratified by model size.
- *Pass/kill:* If Δ(70b) > Δ(27b) (bigger model drops more under perturbation), the raw gap
  *overstates* the capability deficit; the mission target should be the **perturbed** gap.
- *Metric:* perturbation-Δ by model size; perturbed-gap vs raw-gap.

## AT — Miss decomposition (route every FN to its cause)
Impact: High (diagnostic) · Effort: Low · extends AC/AH1/AN1

**Thesis.** The memo names two failure modes (over-annotation, complete miss) but never
*decomposes* a miss. Every FN has exactly one of four causes with four different fixes:
**(a) coverage** — never navigated to the file; **(b) reasoning** — read it, didn't flag;
**(c) scoring** — flagged, matcher didn't match (AR2); **(d) retention** — flagged, a downstream
filter dropped it (AN). These are *deterministically separable* using data already collected: the
AC observation block records files touched; AN1 gives per-stage ρ; AR2 gives the matcher verdict.

**Grounding.** AC captures read-paths; AH1/AN1 propose per-stage ρ; AR2 the matcher. The join
across them — which the envelope never computes — routes each FN automatically.

**AT1 — Per-FN cause attribution.** For each ground-truth vuln missed, classify by the join of
(was its file in the observed read-set?) × (did any stage emit a matching finding pre-filter?) ×
(matcher tolerance). Emit an FN-cause histogram per fixture into the envelope.
- *Test:* Backfill over recent runs — no new model calls (uses stored observations + records).
- *Pass/kill:* Decisive routing: coverage-dominated misses → W6/U ordering & AM min_iter; reasoning
  misses → E/H/skills; scoring misses → AR2; retention misses → AN. The workshop "recall collapse on
  large Django" resolves to a specific cause instead of a vibe.
- *Metric:* FN-cause distribution per fixture; per-cause share of total miss.

## AU — Adopter cost & utility (what "for less" actually costs)
Impact: High · Effort: Low–Med · reframes the objective function

**Thesis.** "For less" is measured in **tokens** throughout, but a self-hoster's real cost is
**GPU wall-clock** and **analyst hours**, and neither tracks token count. Two corrections change
where the A-series and the whole cost story should aim.

**AU1 — Wall-clock, not tokens; and output is co-equal, not 1/47th.** On a self-hosted GPU,
prefill (input) is batched/compute-bound at high tok/s; decode (output) is sequential/
bandwidth-bound at ~10–50× *lower* tok/s per token. So the XBOW **47:1 token ratio (12.67M in /
270k out) is roughly ~1:1 in wall-clock** — and AG4's "input is where the leverage is (≈47×)" is a
token-count artifact, not a latency truth. Worse for that framing: once A1/E5 prefix-caching lands,
repeated prefill is ~free and **output tokens dominate wall-clock outright** — so
generation-side reductions (terser reasoning, shorter records, D4's summarizer) may beat input
compaction on the metric adopters feel.
- *Mechanism:* Measure prefill-tok/s and decode-tok/s on the real stack; convert every token
  figure to GPU-seconds; re-rank the A-series and AG by wall-clock, pre- and post-caching.
- *Test:* A-series/AG changes scored in tokens vs GPU-seconds; input-side vs output-side reductions
  on wall-clock at equal recall.
- *Pass/kill:* Confirm the input:output wall-clock ratio ≪ 47:1 and that post-cache the leverage
  shifts output-side — redirecting where A/AG effort goes. Kill AG4's premise if it holds only in tokens.
- *Metric:* GPU-seconds/run; prefill:decode tok/s ratio; wall-clock Δ input-side vs output-side.

**AU2 — Precision-weighted utility (triage cost).** For a security team the binding cost is
analyst time per finding to review. F1 weights P=R; adopter utility is precision-heavy at scale
(alert fatigue). A config at F1=0.5 emitting 200 findings can cost more human time than one at
F1=0.5 emitting 30. Add a triage-cost metric (Fβ with β<1, or findings-to-review budget) and show
F1-ranked configs diverge from utility-ranked ones.
- *Test:* Re-rank recent runs by Fβ<1 / cost-to-triage; compare to F1 ranking.
- *Pass/kill:* If rankings diverge, the memo's precision-axis directions (C/K/L2/triage) are
  *under*-valued by F1; adopt a documented β. (Ties AC6's precision/recall dial.)
- *Metric:* rank correlation F1 vs Fβ<1; findings-emitted per confirmed TP.

**AU3 — Severity-weighted recall.** All findings count equally today; missing an RCE ≠ missing an
info-leak. A CVSS/impact-weighted recall changes rankings and better reflects value (complements
S2's chain-aware severity, which the scorer doesn't yet honor).
- *Test:* Re-score with severity weights; re-rank.
- *Pass/kill:* Confirm a config strong on low-severity breadth but weak on criticals drops under
  weighting — the adopter-relevant reorder.
- *Metric:* severity-weighted recall; ranking shift vs unweighted.

## AV — Two capability hypotheses the lenses imply but don't test

**AV1 — Position/ordering bias (distinct from AH4 context-length).** AH4 tests recall vs context
*length*; it never isolates *position*. Small models have strong primacy/recency bias
(lost-in-the-middle), so *where* a file/finding/observation sits in the prompt changes whether it's
attended to — at fixed length. This is a separate, cheaper axis and it interacts with A1 (a stable
prefix pins position) and AE (layer order).
- *Test:* Shuffle the order of injected files/skills/observations at fixed total length; measure
  recall variance attributable to position alone, by model size.
- *Pass/kill:* If position-only variance is non-trivial (esp. on the 27b), prompt assembly needs a
  deterministic salience-ordering rule (most-suspect slice last/first), and AC/AE injection order
  becomes a tunable — a near-free recall lever. Kill if recall is position-invariant.
- *Metric:* recall σ from ordering permutations × model size; ties A1 prefix-stability.

**AV2 — Scaling-cliff characterization (foundational F-class).** The workshop "recall collapse on
large Django" is anecdotal; the pipeline's recall-vs-target-size curve is never plotted. Is
degradation smooth (predictable, budget-fixable) or a cliff (needs decomposition)? This *is* the
empirical input AM1's min_iter scaling law presupposes, and it predicts where the pipeline breaks
before an adopter hits it.
- *Test:* Recall/precision/tokens vs target size (LoC, #files, #endpoints, call-graph size) across
  small→large fixtures + synthetic size sweeps (concatenate/prune a fixture).
- *Pass/kill:* Fit the curve; a cliff localizes to a mechanism (context ceiling → W/AH4; coverage →
  U/W6; convergence → AF). Feeds AM1 with data instead of the "large projects need more" heuristic.
- *Metric:* recall(size); knee location; which sub-metric (coverage vs reasoning, via AT1) drives the drop.

---

# Part 2 — Test-scenario improvements to existing directions

**1. Mission-asymmetry: correct the ceiling bias (E1–E5, G3, H2, L3, O3, X4, AA1, AC4, AE2, AH5,
AI, V — the entire small-LLM axis).** `uplift(27b) > uplift(70b)` is confounded: a 70b at 0.90
baseline has ≤0.10 headroom vs the 27b's 0.50, so larger 27b uplift is partly mechanical, not
mechanism. Fixes: (a) report **headroom-normalized** uplift `Δ/(1−baseline)` or gap-closure
fraction, not raw Δ; (b) **void the asymmetry test when the 70b baseline is ceilinged** (>~0.85 on
the fixture) — pick harder fixtures where the large model isn't saturated, or the comparison is
uninformative; (c) prefer the gap metric `recall70−recall27 before/after` (E1/G3/H2 already use it)
but guard it against ceiling-driven shrinkage. This is one edit applied to ~15 hypotheses and it's
the difference between measuring the mission and measuring headroom.

**2. Common-random-numbers / paired seeds (all A/Bs).** The memo re-runs "same fixtures + model
pair" (paired on fixture, good) but doesn't pin the *sampling* seed across arms. Pinning it (CRN) so
arm-A and arm-B see identical decode randomness turns each A/B into a paired-difference test — the
variance-reduction is often 2–5×, letting the *same* pass@N detect *smaller* real effects. Directly
attacks the pass@1-noise problem behind the "quick-wins were noise" finding. Requires a `--seed`
path in `serve.sh` + eval (absent today) — and AR1 first, to know how much determinism the stack
can even deliver.

**3. Sequential early-stop for the A/Bs themselves (meta-apply AM/U to the eval loop).** Running
every arm to a fixed pass@N over-spends on clearly-null and clearly-huge effects. A sequential test
(SPRT / group-sequential) stops each A/B as soon as the effect clears or fails its band — the
eval's own token bill drops with no loss of rigor. The roadmap already argues this for the pipeline
(U3/AM3); apply it to the experiment harness.

**4. N2 — reverted real patches over synthetic injection.** "Inject vulns into diffs" produces
synthetic bugs (P) that may not resemble real introduced ones. Prefer **reversing real security-fix
commits** (SStuBs/CVEfixes/their own N-day corpus): the reverted patch *is* the ground-truth
introduced bug, at zero labeling cost and full realism. Keep synthetic (P) for volume; use reverted
patches for the validity anchor. Shares N's and P's fixture substrate.

**5. Q1 — close the two open holes.** "Small model authors semgrep rules; rule-only recall within X%
of model scan" is (a) **circular without a seen/held-out split** — rules authored against known
examples can encode fixture specifics (same overfit the memo polices in B4/F3), so author on a
golden split, score on held-out; and (b) **uninformative without the AO floor** — the bar isn't
"model-authored rule vs the model," it's "model-authored rule vs semgrep's *default* ruleset." If
authoring doesn't beat the free default, it added nothing. Fold AO1 in as Q1's baseline.

**6. B5 — score above-floor and report the interaction.** `{27b,70b}×{semgrep on/off}` conflates
semgrep-closes-the-gap with semgrep-helps-both. Report the **interaction term** (does semgrep uplift
27b *more* than 70b?), not just the `27b+semgrep ≥ 70b` cell; and credit only **above-floor** recall
(AO) so a "win" isn't semgrep's own deterministic finds re-counted as model capability.

**7. AC3/Z1/Z2 — separate serving noise from "model noise."** These conclude the injected block is
byte-identical ⇒ residual variance is intrinsic model noise. But AR1 shows a *serving* variance term
(batching nondeterminism) that is neither retrieval (Z) nor GT (AP) noise. Re-state the pass
criterion as "observation-block byte-stability holds **and** outcome σ ≤ AR1's serving floor" — else
some of the ±0.07 attributed to model noise is actually attributable, removable serving noise.

---

## Where these plug into the existing priority matrix

- **Do-first (measurement, ~free, protects everything):** AR1 serving floor, AR2 scorer sweep,
  AT1 miss-decomposition backfill, AU1 wall-clock conversion — all backfill over stored runs, no new
  model calls except AR1's repeat sweep. Same tier as F4/F2/AN1/AO1/AP1.
- **Decisive experiments (run before trusting the recall numbers):** AS1 perturbation-sensitivity,
  AS2 public-vs-fresh — if contamination is large, much of the memo's recall re-baselines.
- **Test-scenario fixes:** #1 (ceiling correction) and #2 (CRN seeds) are edits to the shared
  protocol, not new work — apply once, everything downstream tightens.
