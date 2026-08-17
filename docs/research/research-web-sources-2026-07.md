# New literature sweep (2026-07-07) — mapped to research.html directions

Sweep of 2025–2026 papers/blogs/tools relevant to the memo (A–AQ) and the addendum additions
(AR–AV). Each entry: what it is · which direction it touches · **confirm / extend / challenge / impl**.
Sources are the actual search/fetch results from this session.

---

## ★ Top reads (highest relevance)

1. **Thinking Machines — "Defeating Nondeterminism in LLM Inference"** (2025) + **SGLang deterministic
   inference** (LMSYS, 2025-09-22). *Impl for AR1.* Root cause of temp-0 nondeterminism is **batch-size
   variance in kernel reduction order** (RMSNorm/matmul/attention), not GPU concurrency. Batch-invariant
   kernels give **1000/1000 identical runs** on Qwen at ~2× slowdown (SGLang cut it to ~34%). Explicitly:
   *current llama.cpp/vLLM/SGLang cannot guarantee reproducible temp-0 outputs because they adapt kernels
   to batch size — divergence appears once outputs exceed ~100 tokens.* → **This is exactly the AR1
   serving-noise floor, and it names the contractor stack.** Actionable: `-np=1` (AR1's mitigation)
   reduces but doesn't remove it; the real fix is batch-invariant kernels, now shipping in SGLang.
   https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ ·
   https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/

2. **SoK: Measuring What Matters for Closed-Loop Security Agents** (arXiv 2510.01654). *Confirms the whole
   measurement agenda — F/AN + AR/AS/AT/AU.* An external SoK that independently identifies the same
   validity threats: **contamination** (AS), **self-asserted exploitability** (memo C1), **per-stage
   attribution** (AN1 / AT1), **cost opacity** (F2 / AU1), **benchmark specificity/generalization** (AS).
   Its recommendations line up 1:1: mandate independent exploit verification, decompose metrics by phase,
   report resource consumption, rotate datasets to fight contamination, baseline vs human + prior agents.
   → Cite as the peer-reviewed backbone for the memo's F-series and the addendum's AR/AS/AT/AU.
   https://arxiv.org/pdf/2510.01654

3. **CyberGym** (arXiv 2506.02548 · cybergym.io) & **ZeroDayBench** (arXiv 2603.02297). *Fixtures for
   AS / F — contamination-resistant real-vuln benchmarks.* CyberGym = 1,507 historical vulns from 188
   projects; performance **correlates with real-world discovery**; GPT-5 → 22 confirmed zero-days.
   ZeroDayBench = deliberately **unseen** vulns "not present in existing public datasets." → These are
   the honest, low-contamination fixtures AS2 asks for; wire them as held-out corpora (extends F3).
   https://arxiv.org/pdf/2506.02548 · https://www.cybergym.io/cybergym/ · https://arxiv.org/html/2603.02297

4. **VulAgent — Hypothesis-Validation Multi-Agent Vulnerability Detection** (arXiv 2509.11523).
   *Published prior art for AB + L + AH.* Exactly the memo's shape: hypothesis-generation → validation →
   multi-round false-positive reduction, multi-agent, improving both P and R over single-model baselines.
   → The AB→C/AH→L pipeline is validated externally; borrow their validation-loop design.
   https://arxiv.org/pdf/2509.11523

---

## Cluster 1 — Eval validity: determinism, contamination, judge variance (AR / AS / AR2-3)

- **Thinking Machines determinism + SGLang** — see Top-read #1. *impl → AR1.*
- **"LLM Benchmark Datasets Should Be Contamination-Resistant"** (arXiv 2605.19999) and **"When
  Benchmarks Leak: Inference-Time Decontamination"** (2601.19334). *confirm → AS.* Key point matching
  **AS1**: contamination survives **paraphrase/translation/table-reshaping** — it "changes form" and slips
  past string filters. That is precisely why AS1 uses *semantic-preserving perturbation* (rename/relocate)
  as the memorization probe. https://arxiv.org/pdf/2605.19999 · https://arxiv.org/html/2601.19334v1
- **LiveCodeBench** (2403.07974) — *reference → AS/F.* Contamination-limited, time-refreshed code eval;
  the "refresh over time" pattern is the standing-benchmark discipline AS2 implies.
- **"How Much Can We Forget about Data Contamination?"** (OpenReview Pf0PaYS9KG) — *extend → AS3.*
  Quantifies contamination decay/persistence; supports treating contamination as a measurable bias term.
- **Reliability without Validity: LLM-as-a-Judge across Agreement, Consistency, Bias** (2606.19544) and
  **Bias and Uncertainty in LLM-as-a-Judge Estimation** (2605.06939). *extend → AR2/AR3/AV1.* Even though
  contractor's scorer is *deterministic* (so no judge bias), these give the right statistical frame: a
  **variance-component decomposition treating runs × items × position as random effects** → reliability
  coefficients. That is a cleaner backbone than AR1+AR3+AV1 stated separately (adopt generalizability
  theory). Position bias in judges ↔ **AV1** position bias in the worker. https://arxiv.org/html/2606.19544v1
- **AgentProp-Bench: Judge Reliability, Propagation Cascades, Runtime Mitigation** (2604.16706) —
  *extend → AN/AH.* Error propagation across agent stages = the AH cascade / AN retention loss, measured.

## Cluster 2 — Cost, caching, harness>model (A1 / E5 / AU / AH5 / E)

- **llama.cpp `--cache-ram`** (host-RAM prompt caching, added Oct 2025) — *impl → A1/E5, actionable.*
  Restores a slot's KV when a new request shares the prefix (same system prompt / conversation prefix).
  **But `deploy/llama.cpp/serve.sh` sets `CACHE_RAM=0` — the memo's A1 lever is literally disabled in
  your own serving binary.** Combined with the A1 addendum's prompt-prefix-byte-stability requirement,
  this is a config flip + prompt-order fix, not new infra. https://jessequinn.info/blog/llama-cpp-cache-ram-prompt-caching
- **vLLM Automatic Prefix Caching** (docs.vllm.ai) — *reference → A1.* Mature block-hash prefix reuse
  (incl. cross-model); the alternative serving path if llama.cpp caching proves too manual.
- **CacheWise: KVCache management for serving LLM coding agents** (2606.16824) — *extend → A1/AG4/AU1.*
  Workload-aware KV management for agent traffic specifically; informs AU1's wall-clock cost model.
- **"The Agent Harness: Why the LLM Is the Smallest Part"** (MongoDB, 2026) + **Holistic Agent
  Leaderboard** — *confirm → AH5/E.* Concrete harness-only wins: Vercel 80→100%, LangChain bottom→top-5,
  Harvey 2× — direct evidence for "capability = harness × model," the memo's E/AH5 thesis.
- **AgentTTS** (2508.00890), **compute-optimal test-time scaling / "1B can beat 405B"**, **MetaScale**
  (2503.13447), **"Generalizing test-time compute-optimal scaling as an optimizable graph"** (2511.00086)
  — *extend → E/AM/AL.* Test-time compute as the small-model lever; the "optimizable graph" framing ties
  AD (declarative composition) to AM (budget sizing).
- **"Towards a Science of AI Agent Reliability"** (2602.16666) — *reference → F/AR.*

## Cluster 3 — Static-analysis hybrids & rule authoring (Q / B5 / H / AO)

- **LLM-based multi-agent system for generation & evolution of CodeQL rules (C/C++)** (ResearchGate
  403441399) — *confirm → Q1/Q2.* Multi-agent authors *and evolves* CodeQL rules — the compounding
  rule-library of Q2, realized. Note their caveat (LLM-as-standalone-detector = high recall, many FPs)
  motivates the hybrid, i.e. AO floor + model residual.
- **Hybrid LLM+SAST (Semgrep/CodeQL + LLM triage)** — *impl numbers → B5/AO/Q3.* Reported **2.5×
  detection, −91% false positives** for deterministic-engine + LLM-triage pipelines. Concrete Pareto
  evidence for the Q3 hybrid and the AO floor-then-residual design.
  https://www.sciencedirect.com/org/science/article/pii/S1546221826000603 · https://science.lpnu.ua/ictee/all-volumes-and-issues/volume-6-number-1-2026/sast-improvements-using-llm-cicd-pipelines
- **"Can LLM Prompting Serve as a Proxy for Static Analysis?"** (2412.12039) — *challenge/scope → B/AO.*
  Sets when prompting substitutes for SA vs when the deterministic floor is needed — sizes AO's value.
- **8 AI SAST Tools 2026** (Augment) / **Semgrep vs CodeQL 2026** (Konvu) — *reference → AO.* Current
  tool landscape and default-ruleset baselines for the AO floor.

## Cluster 4 — Hypothesis-gen, multi-agent, root-cause reasoning (AB / L / R / B / E)

- **VulAgent** — Top-read #4. *confirm → AB/L/AH.*
- **VulInstruct: Root-Cause Reasoning via Security Specifications** (2511.04014) — *extend → R/AQ/B.*
  Teaches root-cause reasoning from *security specifications* — the codified-intent substrate R wants and
  AQ2 mines from tests. Strong pairing for the intent/invariant direction.
- **VULPO: Context-Aware Vulnerability Detection via On-Policy LLM Optimization** (2511.11896) —
  *adjacent/out-of-scope → B/E.* Shows context framing drives detection, but via RL fine-tuning (the
  memo's no-finetune boundary) — cite as the training-based counterfactual to the harness-only thesis.
- **MAS-PromptBench: When Does Prompt Optimization Improve Multi-Agent Systems?** (2606.23664) —
  *challenge → AK/L.* Caution: prompt optimization doesn't always help multi-agent systems — guards AK
  against assuming GEPA-style gains transfer to the planner+worker ensemble.

## Cluster 5 — Prompt/skill/task auto-refinement (AK)

- **GEPA — Reflective Prompt Evolution** (arXiv 2507.19457, **ICLR 2026 oral**; DSPy tutorial;
  gepa-ai/gepa repo) — *impl → AK1.* +20% vs GRPO at **35× fewer rollouts**, +13% vs MIPROv2, **measured
  on Qwen3-8B** (small-model, on-mission), Pareto candidate pool from execution traces. The memo already
  cites GEPA; now it's an available library with an ICLR-oral result on the exact model class.
  https://arxiv.org/pdf/2507.19457 · https://dspy.ai/tutorials/gepa_ai_program/ · https://github.com/gepa-ai/gepa
- **Maestro: Joint Graph & Config Optimization for Reliable AI Agents** (2509.04642) — *extend → AK/AD.*
  Optimizes the agent *graph* and configs jointly — AK applied to composition (AD), not just prompts.
- **TextBO** (2511.12063), **optimize_anything** (2605.19633), **MemPro: Agentic Memory as Evolvable
  Programs** (2606.00619) — *extend → AK/O.* Text-space Bayesian opt (eval-efficient — pairs with AR3
  multiplicity), universal text-param optimizer, and memory-as-evolvable-program (O experience memory).

## Cluster 6 — N-day / patch-diff / self-play (N / P / O / C)

- **Patch-to-PoC: Agentic LLM for Linux Kernel N-Day Reproduction** (2602.07287) — *impl → N/C, and
  refinement #4.* Systematic study of patch→PoC agents; **studies how the LLM knowledge-cutoff affects
  reproduction** — a direct contamination signal (ties AS). Validates N2's reverted-patch-as-ground-truth
  design over synthetic injection. https://arxiv.org/pdf/2602.07287
- **Team Atlanta — "Patching Vulnerabilities with Coding Agents 2026"** (AIxCC ensemble writeup) —
  *reference → whole thesis / P.* DARPA AIxCC context; AIxCC synthetic vulns (C/Java) are a self-play/
  synthetic-fixture corpus (P). https://team-atlanta.github.io/blog/post-patch-2026-ensemble/
- **EvoRepair: Experience-Based Self-Evolution for Repair Agents** (2605.30105) — *extend → O/P.*
  Repair agents that self-evolve from accumulated experience — the O (experience memory) + P (self-play)
  loop, on the repair side.
- **StriderSPD (binary security-patch detection)**, **curriculum preference optimization w/ synthetic
  reasoning** — *extend → N/P.* Synthetic + patch-based signal generation.

## Cluster 7 — Search / planning / value-of-information (AL / S / U)

- **ToolTree: Dual-Feedback MCTS + Bidirectional Pruning** (2603.12740) — *extend → AL/U.* ~10% over SOTA
  planning; pruning = the VoI-style budget control U wants over the plan tree.
- **PAC-MCTS: Bias-Aware Pruning for Robust LLM-Guided Search** (2604.14345) — *extend → AL6.* Prunes
  biased branches — directly the "refuted-path memory prevents entrenchment" idea (AL6), formalized.
- **Tree-GRPO: Tree Search for LLM Agent RL** (2509.21240, ICLR 2026; AMAP-ML repo) — *adjacent → AL.*
  Tree-structured rollouts over ReAct step-nodes (RL-based; note the no-finetune boundary).
- **LATS / Agent Q** (via survey) — *reference → AL/AH.* Self-reflection into MCTS + offline training over
  success/fail trajectories — the memo already leans on these; still the canonical pattern.

## Cluster 8 — Access-control / BOLA / business logic (R / E3 / AQ / G)

- **"Broken Object Level Authorization in the Wild: Taxonomy from 100+ Bug-Bounty Disclosures"**
  (2605.25865) — *fixture/knowledge → G/AA/R.* Empirical BOLA taxonomy (84/107 confirmed in-scope) — a
  real knowledge source to fill the R/AA missing-authz cell and a candidate fixture generator.
  https://arxiv.org/html/2605.25865
- **"Are There IDORs Lurking in Your Code? LLMs Are Finding Critical Business-Logic Vulns"** (Security
  Boulevard, 2026-01) — *confirm → R/E3.* Industry signal that LLMs now surface business-logic/IDOR at
  scale. https://securityboulevard.com/2026/01/...
- **Hybrid BOLA detection** (CSIT 2025 IC_10) + **StackHawk/APIsec multi-user testing** — *confirm →
  E3/AQ2.* Reinforces the core mechanic: BOLA needs **multiple authenticated users** (single-session
  scanners miss it) — exactly E3's two-user probe and AQ2's mined-ownership-assertions.
- **OWASP Top-10 for Business Logic Abuse (2025)** — *reference → R/G.* New taxonomy to align skill cells.

---

## Actionable now (config/wiring, not research)

- **Flip `CACHE_RAM=0` → >0 in `deploy/llama.cpp/serve.sh`** and stabilize prompt-prefix byte-order
  (A1 addendum) — the prompt-caching lever is disabled in your own binary. (A1/E5)
- **Adopt SGLang batch-invariant kernels (or pin `-np=1`) for eval runs** to get the AR1 serving floor
  toward reproducible; measure the residual σ either way. (AR1)
- **Wire CyberGym + ZeroDayBench as held-out, low-contamination fixtures** (AS2/F3); run
  perturbation-sensitivity (AS1) on the existing public fixtures first — it's free.
- **Try GEPA (gepa-ai/gepa) on the trace/scan worker prompt** — ICLR-oral, Qwen3-8B, 35× cheaper than RL;
  the memo's AK1 now has a drop-in library. (AK)
- **Adopt the hybrid floor-then-residual pattern** (Semgrep default ruleset as AO floor; LLM only credited
  above it; optionally LLM-authored rules per Q) — external hybrids report 2.5× detection / −91% FP.

---

# LLM-based vulnerability discovery — dedicated deep-dive (2026-07-07)

The single most useful pointer: **Awesome-LLMs-for-Vulnerability-Detection**
(https://github.com/huhusmang/Awesome-LLMs-for-Vulnerability-Detection) — continuously-updated index across
function-level / repository-level / agentic / smart-contract detection + datasets, benchmarks, surveys.
Use it as the standing tracker instead of re-searching.

## A. Reality-check benchmarks (temper the thesis; strengthen AO / AS / F / AR)

- **"LLM-based Vulnerability Detection at Project Scale: An Empirical Study"** (arXiv 2601.19239, Jan
  2026). *challenge → whole detection thesis; motivates AO.* At project scale, LLM recall is only **~21%
  (C/C++) / ~34% (Java)**, and even the best tool has an **85.3% false-discovery rate**. → The field-wide
  honest baseline; contractor's F1=.55–.65 on small fixtures looks strong partly *because* the fixtures
  are small and public (AS). Cite as the sobering baseline the memo's honesty slide (AH) gestures at.
  https://arxiv.org/pdf/2601.19239
- **"LLM-based Vulnerability Discovery through the Lens of Code Metrics"** (ICSE 2026;
  mlsec.org/docs/2026-icse.pdf). *challenge → E/B; strong argument for AO.* A classifier trained **solely
  on code metrics performs on par with SOTA LLMs** — and "progress has stalled." → The deterministic
  floor (AO) may match the LLM on much of the surface; the LLM budget belongs on the floor-invisible
  residual (Q3 hybrid). The most pointed challenge to "just scan with the model." https://mlsec.org/docs/2026-icse.pdf
- **"Are Frontier LLMs Ready for Cybersecurity? Dual-Mode Vulnerability Benchmarks"** (2605.23243).
  *fixture + caveat → F/AS/C.* **5 production-style web apps, 118 ground-truth vulns, 20+ CWE families**;
  frontier LLMs show **10–50% FP, 4–8% black-box coverage, 2–3 refusals per 5 legitimate offensive runs.**
  → A ready-made web-app fixture set matching contractor's target profile; the refusal rate is a real
  serving concern for offensive tasks. https://arxiv.org/html/2605.23243v3
- **SEC-bench Pro** (2605.26548) — *reference → F.* Long-horizon software-security task benchmark (the
  multi-step regime the memo's planner targets).
- **"Everything You Wanted to Know About LLM-based Vuln Detection But Were Afraid to Ask"** (2504.13474)
  — *survey → orientation.* Broad SoK of methods/pitfalls.

## B. FP-reduction / verify-for-precision — the AH funnel, realized (AH / K / L2 / AO / AP)

- **Refute-or-Promote: Adversarial Stage-Gated Multi-Agent Review for High-Precision Defect Discovery**
  (2604.19049). *confirm → L2 + AH + AP.* Almost the memo's exact design: adversarial stage-gates that
  **refute or promote** each candidate — L2's majority-refute panel + AP's FP adjudication in one loop.
  Closest external match to the memo's verification funnel. https://arxiv.org/pdf/2604.19049
- **ZeroFalse: Improving Precision in Static Analysis with LLMs** (2510.02534). *impl → AH/K/AO.* LLM as
  a precision filter over static-analysis output; **grok-4 F1=0.912, gemini-2.5-pro F1=0.910, precision
  >0.85 on OWASP.** Concrete evidence for "generate cheap with SA, filter for precision with the LLM."
  https://arxiv.org/html/2510.02534
- **"Reducing False Positives in Static Bug Detection with LLMs" (LLM4PFA)** (2601.18844). *impl →
  AH/K.* Accuracy 0.93–0.94 across backbones — the filter-not-generator thesis (AH) with numbers.

## C. Repository-scale agentic discovery — the memo's H/I/W/Y, realized (H / I / W / E / Y)

- **Codebadger** (via project-scale search). *confirm → H + I + E.* High-level **program slicing, taint
  tracking, dataflow, semantic navigation** so the agent explores a large repo without exhaustive reads;
  navigated an **8,000-method codebase**, found a real **libtiff buffer overflow**, and patched
  **libxml2 CVE-2025-6021 first try.** → The H (dataflow/taint primitive) + I (graph nav) + E
  (offload-reasoning-to-tools) thesis, finding real bugs. Strongest external validation of H.
- **VulnLLM-R: Reasoning LLM with Agent Scaffold** (2512.07533). *confirm → I1 + H.* Builds a **call
  graph from CodeQL direct calls augmented with type-based indirect-call resolution** — precisely the
  interface/DI-dispatch resolution I1 says contractor loses on remote/overlay FS. A concrete design for
  the indirect-call gap. https://arxiv.org/pdf/2512.07533
- **Revelio: Cost-Efficient Agentic Memory-Safety Detection for Repository-Scale Codebases**
  (2606.22263). *confirm → A/W/AG.* "Cost-efficient" + "repository-scale" = the memo's exact
  cost×context thesis; read for its budget/navigation tactics. https://arxiv.org/pdf/2606.22263
- **Bridging Code Property Graphs and Language Models for Program Analysis** (2603.24837). *extend →
  I/H.* CPG (a richer superset of contractor's call graph — adds data/control-dependence edges) as the
  LLM's structured substrate. The natural upgrade path for Direction I. https://arxiv.org/html/2603.24837v1
- **FuzzingBrain V2: Multi-Agent Discovery + Reproduction** (2605.21779) and **Agentic Fuzzing**
  (2605.10074) + **FirmAgent** (NDSS 2026, fuzzing-assisted LLM agents). *extend → T.* The grey-box /
  fuzzing-guided direction (T), with reproduction (C) attached.
- **RAVEN: Agentic RAG for Automated Vulnerability Repair** (2606.22647). *extend → AJ/O.* RAG-over-a-
  store for repair — AJ's disciplined-RAG applied downstream of discovery.

## D. Real-world milestones (context for the mission)

- **Google Big Sleep found CVE-2025-6965 (SQLite) in the wild** — reportedly the first AI agent to foil
  an in-the-wild exploit before use (Google "summer of security" blog).
  https://blog.google/innovation-and-ai/technology/safety-security/cybersecurity-updates-summer-2025/
- **XBOW** — #1 on HackerOne (June 2025), **1,060+ valid submissions**, $155M Series C (Mar 2026),
  co-founded by Oege de Moor (GitHub Copilot). Contractor already uses XBOW benchmarks; this is the
  frontier the small-LLM mission is trying to approach at 1/Nth the cost.
- **Anthropic Mythos Preview (Apr 2026)** — reported thousands of high-severity vulns across major
  OSes/browsers (frontier-scale discovery; the capability ceiling the harness aims to reach with small
  models). *appsecsanta "AI Pentesting Agents 2026" tested 39+ tools — a landscape scan for L/V ensemble.*

## Takeaways specific to LLM-based vuln discovery

1. **The honest field baseline is low** (21–34% recall, 85% FDR at project scale; code-metrics ties LLMs).
   This *strengthens* the memo — it means (a) the deterministic floor (AO) is not a formality but a real
   competitor, (b) precision via verification (AH/K/L2/ZeroFalse/Refute-or-Promote) is where the field is
   actually winning, and (c) contractor's higher fixture F1 must be contamination-checked (AS) before it's
   believed to beat the field.
2. **Repo-scale is solved by tools, not context** — Codebadger/VulnLLM-R/Revelio all externalize taint +
   call-graph + slicing (the memo's H/I/E), confirming the offload thesis with real CVEs.
3. **The verify funnel is the consensus winning pattern** — Refute-or-Promote and ZeroFalse independently
   arrive at generate-cheap → filter-for-precision (the memo's AH), the same shape as vuln_scan_trace.

## What challenges the memo

- **MAS-PromptBench**: prompt optimization doesn't reliably help multi-agent systems — temper AK's
  expected transfer to the planner+worker ensemble.
- **Determinism cost**: full temp-0 reproducibility costs ~34–100% throughput — AR1's floor is real but
  *pinning* it isn't free; may be an eval-only mode, not a production default (ties AU1 wall-clock).
- **Training-based frontier** (VULPO, Tree-GRPO, CTF-Dojo, Cyber-Zero) keeps posting gains via RL/SFT —
  a standing pull against the strict no-finetune stance; worth a periodic "is the boundary still right?"
  re-check as adopter tooling for cheap LoRA matures.

# Update scan — 2026-08-05

## Difficulty-aware agent control (AX)

- **PentestGPT v2 / “What Makes a Good LLM Agent for Real-world Penetration Testing?”**
  (arXiv 2602.17622). Separates tool/knowledge failures from search-complexity failures, adds typed
  interfaces for 38 tools, Task Difficulty Assessment, an external Evidence-Guided Attack Tree Search,
  and persistent state outside conversational context. Its Tool Layer ablation raises XBOW completion
  54%→68%. https://arxiv.org/abs/2602.17622
- **APT-Agent** (arXiv 2605.24949). Hybrid command rectification plus command-specific memory; reports
  84.29% end-to-end exploitation on seven Metasploitable 2 services under its evaluation. Useful for
  AX1's capability-failure recovery arm, though the target set is small. https://arxiv.org/abs/2605.24949
- **PenForge** (ICSE-NIER 2026; arXiv 2601.06910). Constructs expert agents on demand after
  reconnaissance rather than using a fixed specialist roster; reports 12/40 CVE-Bench zero-day-mode
  successes. https://arxiv.org/abs/2601.06910

## Ecologically valid evaluation (AY)

- **Autonomous penetration capability benchmark** (arXiv 2606.13079). Creates 300 target servers and
  varies secure-service distractors beside the vulnerable service (one vs three), providing a direct
  template for measuring recall degradation with distractor density. https://arxiv.org/abs/2606.13079
- **CyberGym-E2E** (ICLR 2026 Agents in the Wild Workshop). Reproducible OSS-Fuzz build environments;
  evaluates discovery, PoC, patching, and functionality consecutively. The paper reports the dataset
  expanding to 920 tasks. https://openreview.net/pdf?id=bffc0196f77135bed3db638acbab2c0d95836b03
- **GitHub Security Lab Taskflow Agent** (open source, 2026). Declarative staged taskflows with schema
  validation, retries/checkpoints, and CodeQL-derived context. Published auditing guidance explicitly
  allows a no-vulnerability conclusion and requires file/line evidence.
  https://github.com/GitHubSecurityLab/seclab-taskflow-agent
- **Microsoft MDASH / MAI-Cyber-1-Flash** (official Microsoft reports, May–July 2026). Microsoft reports
  88.45% then 96% on CyberGym and nearly 50% cost savings after introducing its sparse cyber model.
  Treat these as vendor-reported system results, not directly reproducible evidence; the testable idea
  is routing most work to a small active model and reserving expensive capacity for the hard residual.
  https://www.microsoft.com/en-us/security/blog/2026/05/12/defense-at-ai-speed-microsofts-new-multi-model-agentic-security-system-finds-16-new-vulnerabilities/
  https://blogs.microsoft.com/blog/2026/07/27/rethinking-security-for-the-age-of-ai/
