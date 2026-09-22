# crawl2action: plan for an end-to-end pipeline from e-commerce crawling to a post-trained LLM

## Context
The `crawl2action` repo is empty apart from a README and LICENSE. The goal is one pipeline that:
1. crawls product catalogs from popular e-commerce sites in SE, UK, ES, US and CA using **Firecrawl**, plus **UCP catalog endpoints** and open datasets;
2. turns that data into training and eval sets for **pre-checkout recommendations** (search/browse/PDP), **post-checkout recommendations** (cross-sell, replenishment), **text generation** (product descriptions, titles, ads, localized copy) and **image generation** (product and lifestyle visuals);
3. post-trains a **student LLM** on **Tinker** (Thinking Machines; Modal as the self-run fallback) with the current recipe (cold-start SFT → reward modelling with human preferences, rubrics and GenRM → GRPO-family RL → on-policy distillation) for now, with **Applied Compute AC2** as a later drop-in behind the same trainer interface;
4. serves the models on **Modal**.

Decisions confirmed with the user:
- **Two model tiers:**
  - The student is **Qwen3.8-27B** (dense, multimodal, Aug 2026). It is the model that gets post-trained and served in production on 1–2 H100/H200. **Inkling-Small** competes with it in the bake-off.
  - The teacher and judge is **Kimi K3** (2.8T MoE, 104B active), GLM-5.3 or Qwen3.8-Max, called through a hosted API. It is used only for generating synthetic data and for grading, never for serving.
- **Images:** the LLM writes structured image briefs, and a separate image model renders them. That model is **Qwen-Image-2.1** (7B, generation plus editing, 10 reference images), served on Modal. An optional LoRA can be trained on crawled product photos.
- **Data sources:** a mix of popular regional stores, open datasets and UCP catalog access. Every source goes through robots.txt and ToS gating.

## Architecture (six stages, each a Modal app plus a `c2a` CLI command)
```
discover → crawl/ingest → normalize+dedupe → build datasets → post-train (Tinker; Modal fallback; AC2 later) → eval → serve (Modal)
                                                                  ↑                                     │
                                                                  └─────── production feedback logs ─────┘
```
Key design choice: the model does **not memorize the catalog**. At serving time, recommendations work as *retrieve then rerank and explain*:
1. an embedding index returns about 50 candidates;
2. the LLM ranks them, writes the explanation and enforces constraints.

Training therefore teaches the model to reason over candidate sets. That makes it robust to catalog churn and lets graders check hallucinations (every returned product ID must be in the candidate set).

## Repo layout (Python 3.12, `uv`, `pydantic`, `typer`, `ruff`, `pytest`)
```
src/c2a/
  schemas.py            # Product, Variant, Offer, Review, Store, Session, TaskExample (pydantic; aligned to UCP catalog + schema.org Product)
  sources/
    registry.yaml       # stores: domain, country, locale, platform, tier, tos_status, method
    discover.py         # Firecrawl /search + platform fingerprinting (Shopify, WooCommerce) + probe /.well-known/ucp
    shopify.py          # /products.json pagination (cheap, structured, no LLM extraction)
    ucp.py              # UCP catalog capability client (REST/MCP search + get product/variants)
    firecrawl.py        # /map + /crawl + /extract (JSON schema = schemas.Product) with rate limits and caching
    open_datasets.py    # Amazon Reviews 2023 (co-purchase and user histories), H&M Personalized Fashion, RetailRocket, Diginetica
    compliance.py       # robots.txt check, ToS allow/deny list, PII stripping, per-domain QPS
  normalize/            # currency to minor units + FX, taxonomy mapping (Google Product Taxonomy), language ID, near-dup (MinHash + image pHash)
  index/                # multilingual embeddings (e.g. Qwen3-Embedding) → LanceDB on Modal Volume
  datasets/
    tasks/              # one module per task family (see below); each emits SFT + RL prompts + eval items
    teacher.py          # teacher API client (Kimi K3 / Qwen-Max), batch, cache, cost cap
    splits.py           # held-out splits by store AND by time to avoid leakage
  graders/              # programmatic + LLM-judge graders shared by eval and RL
  decide/               # decide(evidence, question, labels) -> probs; Decision-1.0-style calibrated classifiers
  train/
    base.py             # Trainer interface: submit(dataset, config) -> run_id; export(run_id) -> HF/LoRA weights
    tinker/             # PRIMARY: SFT, RLHF, RL (custom Rank-GRPO/DAPO losses), OPD loops on Tinker
    envs/               # Modal-hosted rollout envs + reward endpoints called from Tinker loops
    sft.py              # fallback: TRL cold-start SFT / DPO baselines (Modal)
    reward/             # preference-labelling app, rubric generation, GenRM training
    rl/                 # fallback: verl on Modal: GRPO/DAPO/GSPO losses, Rank-GRPO advantage, TIS correction
    opd.py              # on-policy distillation + self-distillation
    ac2/                # LATER: AC2 project (added once access is granted)
  eval/                 # Inspect-based harness: public benchmark adapters + C2A-Bench tracks → reports
  bench/                # C2A-Bench builder: tracks, gold-set sampling, contamination checks, canary
  serve/
    llm.py              # Modal + vLLM, OpenAI-compatible, --enable-lora for per-task adapters
    image.py            # Modal + diffusers Qwen-Image-2.1 (+ optional product LoRA)
    gateway.py          # thin FastAPI on Modal: /recommend, /cross-sell, /generate/text, /generate/image, /feedback
    router/             # vLLM Semantic Router config: signals, decisions, model pool
  cli.py                # c2a discover|crawl|build|train|bench|eval|deploy
configs/                # per-run YAML (model, tasks, mix ratios, budgets)
tests/
```
Storage: raw crawl JSON → parquet on a Modal Volume, with an optional S3/R2 mirror. Secrets (`FIRECRAWL_API_KEY`, `TEACHER_API_KEY`, `AC2_*`, `HF_TOKEN`) live in Modal Secrets.

## 1. Discovery and crawling
**Source tiers** in `registry.yaml`:
- **A. Structured / permissioned (preferred):** UCP-enabled merchants (probe `/.well-known/ucp` and use the catalog capability), Shopify `/products.json`, and open datasets.
- **B. Firecrawl `/extract`:** regional stores whose robots.txt and ToS permit crawling.
- **C. Marketplaces** (Amazon, Zalando, ASOS, El Corte Inglés, etc.): **off by default**. Enable only with explicit ToS clearance. Behavioral signal comes from open datasets instead.

**Seed candidates.** `discover.py` verifies platform, UCP support and ToS, then expands the list automatically by running Firecrawl `/search` for queries like "best online stores {country} {category}".
- **SE:** Happy Socks, Djerf Avenue, CDLP, Apotea, Lyko, Elgiganten, CDON
- **UK:** Gymshark, Represent, Huel, Oliver Bonas, Argos, John Lewis
- **ES:** Hawkers, Scalpers, Mr Wonderful, PcComponentes, El Corte Inglés
- **US:** Allbirds, Glossier, Brooklinen, Bombas, Best Buy, Target
- **CA:** Kotn, Frank And Oak, Knix, Tentree, Peace Collective, Canadian Tire

Target about 50 stores in 6–8 verticals (apparel, beauty, electronics, home, grocery/health) and about 1–2M products. Locales are sv-SE, en-GB, es-ES, en-US, en-CA and fr-CA.

**Crawl mechanics:**
- incremental crawling with ETag and content hash;
- a per-domain QPS cap and a Firecrawl credit budget per run;
- a PII scrub on reviews;
- product images are stored (for the image LoRA and multimodal later) with source and licensing metadata.

## 2. Task families and datasets
| Task | Input | Output | Signal / grader |
|---|---|---|---|
| Pre-checkout rec | user query or browse context, locale, constraints, 50 candidates | ranked IDs + short reasons (JSON) | held-out next-item from open datasets (NDCG@10, Recall@10); constraint satisfaction (price, size, stock, locale); ID-in-candidates |
| PDP "similar / alternatives" | anchor product + candidates | ranked IDs + reasons | same-category/attribute agreement; judge |
| Post-checkout cross-sell | cart/order + candidates | complementary IDs (not substitutes) | co-purchase ground truth (Amazon 2023 / H&M); substitute penalty |
| Replenishment / follow-up message | order history | timing + message | judge + rubric |
| Product copy | product attributes (+ brand voice) | title, description, bullets, SEO meta, ad copy | **attribute-faithfulness check** (every claim traceable to attributes), length/format, language ID matches locale, judge for fluency |
| Localization | copy + target locale | localized copy | language ID, attribute preservation, judge |
| Image brief | product + use case (hero, lifestyle, banner) | structured JSON prompt (scene, style, aspect, reference image IDs, negative prompt) | schema-valid, product fidelity judge on rendered output (VLM judge) |

- SFT data comes from teacher-distilled answers, filtered by the same graders (rejection sampling).
- RL prompts are the same tasks with ground truth attached to the graders.
- Frozen eval sets are held out by **store** and by **time** and are versioned.

## 2b. Decision layer (inspired by vLLM Semantic Router and the Decision-1.0 models)
**What we borrow**
- vLLM Semantic Router (Apache 2.0) composes cheap *signals* into routing *decisions* in front of several models:
  - heuristics such as keywords, language and context length;
  - neural classifiers for domain, PII, jailbreak attempts and complexity;
  - embedding similarity.
- Its **Decision-1.0** models (Lux-9B on Qwen3.5-9B; Nox-4B, Sol-2B; the earlier Kev-9B) take *evidence + a question + labels defined at request time* and return a decision **with probabilities**.
- Two ideas carry over:
  1. **Framing sub-decisions as small, calibrated classification calls** instead of free-form generation.
  2. **A mixture-of-models router** in front of the serving tiers.

**Where decision calls go in crawl2action.** One `decide(evidence, question, labels) -> {label: prob}` interface in `src/c2a/decide/`:
| Decision | Labels (set at request time) | Used for |
|---|---|---|
| Query↔product relevance | exact / substitute / complement / irrelevant (ESCI) | Candidate filtering, pre-checkout reranking features, and the reward for Rec-Pre |
| Cross-sell type | complement / substitute / unrelated | Post-checkout recommendations. A substitute penalty guards against suggesting an item the customer just bought |
| Claim faithfulness | supported / contradicted / not-in-attributes (per claim) | Copy grader and reward. The **probabilities give a denser, calibrated reward** than a 1–10 judge score |
| Shopper intent / stage | browse / compare / buy-now / post-purchase / support | Choosing the recommendation strategy, and routing |
| Constraint check | satisfies / violates, for each price, size, stock or locale constraint | Hard gate at serving time, and a reward penalty |
| Safety / compliance | PII / unsafe product / off-policy content | Crawl scrubbing, and output guard at serving time |

**Model**
- **M2 bake-off for the decider:**
  - Decision-1.0-Lux-9B as it ships (baseline only; verify its license and model card first);
  - the same model fine-tuned on our labels (ESCI and C2A gold subset, via Tinker);
  - a small same-family model distilled from the teacher.
- Metrics: weighted accuracy, **calibration (ECE and Brier score)** and latency.
- The winner does three jobs:
  - a **fast grader and reward model** in the RL loop (a better fit than a free-form judge for any grader that is really a classification);
  - a **serving-time guard**;
  - a **router signal**.

**Serving router.** vLLM Semantic Router (Envoy ext_proc) sits in front of the Modal endpoints. It routes each request, based on signals and decider outputs, to:
- the fast reranker (high-QPS recommendations);
- Qwen3.8-27B, with reasoning turned on only when the complexity signal calls for it (the router's "When to Reason" idea);
- the image pipeline.

PII and jailbreak signals are blocked at the edge. This replaces the hand-written dispatch in `serve/gateway.py`, which is kept only as a thin API layer.

**Benchmark.** C2A-Bench gains a **Decide track**:
- Covers all the decision types above, per locale.
- Scored with weighted accuracy per category (modelled on how the Decision-1.0 models are evaluated) plus ECE/Brier.
- Also measures router quality: cost and latency saved at equal quality, compared with sending everything to the 27B model.

## 3. Post-training (state of the art as of Sep 2026)
The design follows the **sparse-to-dense reward principle** (Microsoft Research, May 2026):
- Spend **sparse, outcome-level reward** (RL) on the model best able to explore with it.
- Pass the resulting behaviour to the deployed model through **dense, token-level supervision**, meaning on-policy distillation (OPD).
- Run student-side RL only after that distillation step.

Qwen3, GLM-5 and MiMo all use OPD in their post-training pipelines. Running plain GRPO directly on a cold student wastes the labelled signal on the least-prepared policy.

**Every model we serve is post-trained by us.** Off-the-shelf checkpoints are only starting points and bake-off baselines.
| Model | How we post-train it (Tinker; Modal fallback) |
|---|---|
| Student (Qwen3.8-27B or Inkling-Small) | Cold-start SFT → RLHF / RL with rubric, Rank-GRPO and verifiable rewards → optional self-distillation |
| Fast reranker | On-policy distillation from the RL-trained student → optional student RL |
| Decider (starts from Decision-1.0-Lux-9B or a same-family small model) | SFT on ESCI plus C2A gold labels → RL with a calibration-aware reward (log-loss / Brier score) |
| GenRM judge | SFT on teacher and human verdicts → RL against human preference labels |
| Image model (Qwen-Image-2.1) | Product LoRA → preference optimization (Diffusion-DPO or Flow-GRPO) |

**Model roles (checked against releases as of 22 Sep 2026)**

The final choice between the two student candidates comes from a **model bake-off in M2**: every candidate runs on C2A-Bench (section 4) before any training money is spent.

| Role | Primary pick | Alternative in the bake-off | Why |
|---|---|---|---|
| Frontier teacher / data generator / rubric writer | **Kimi K3** (2.8T MoE, 104B active, Jul 2026) | GLM-5.3 (top open model on the Artificial Analysis index, Sep 2026), Qwen3.8-Max (2.4T) | Best quality. It is only used through text outputs (API), because a different tokenizer rules out token-level OPD |
| **Main student (RL'd, served)** | **Qwen3.8-27B** (dense, natively multimodal: image and video input, 262K context, Apache 2.0, weights released 13–14 Aug 2026) | **Inkling-Small** (Thinking Machines, 12B active MoE, multimodal, Apache 2.0, Jul 2026) | Qwen3.8-27B replaces the older Qwen3.8-27B. It reads product photos directly and fits on 1× H100 in FP8. Inkling-Small is native to Tinker, and its big sibling Inkling (975B, 41B active) can act as a same-tokenizer OPD teacher entirely inside Tinker |
| Same-family OPD teacher | The RL'd Qwen3.8-27B (or Qwen3.8-Max, if it is available on Tinker) | Inkling, if Inkling-Small wins the bake-off | Token-level distillation needs the same tokenizer |
| Fast reranker (optional) | A small model from the same family as the student, distilled with OPD | – | High-QPS `/recommend` |
| Reward model / judge | A small same-family model trained as a generative reward model (GenRM) | – | Cheap reward for RL, calibrated against Kimi K3 and human labels |
| Image generation | **Qwen-Image-2.1** (7B, generation and editing, Sep 20 2026) | FLUX.2 (highest raw image quality; check its license for commercial use) | Bake-off on product fidelity and text rendering |

**Stages** (every stage is a separate, resumable Tinker or Modal job that writes a versioned checkpoint and an eval report)
1. **Cold-start SFT (off-policy distillation).**
   - Train on Kimi K3 outputs that passed the graders (rejection sampling), with a short reasoning trace plus the final JSON answer.
   - The goal is a formatted, grounded starting policy rather than peak quality.
   - LoRA with high rank, or full fine-tuning if the budget allows.
2. **Reward modelling (the RLHF part).**
   - **Human preferences:** a small labelling UI (a Modal web app) collects pairwise comparisons and rubric scores on copy, recommendation explanations and images. Raters are internal, with native speakers for sv, es and fr.
   - **Rubrics as Rewards:** Kimi K3 writes a *per-instance rubric* for each prompt, grounded in product attributes, locale, brand voice and constraints. The judge scores each rubric item. This gives a reward for tasks that have no single correct answer.
   - **GenRM:** a small Qwen is trained (SFT, then RL) to produce a rubric-grounded verdict. It is validated against held-out human labels, and the target is ≥80% agreement before it is used as a reward.
   - **Implicit feedback:** production clicks, add-to-cart and purchase events feed into the reward-model data. Recommendation changes are checked offline with counterfactual evaluation (IPS/DR estimators) before any A/B test.
3. **RL on the 27B (RLVR plus rubric rewards).** This uses the modern GRPO family:
   - **Loss fixes:** DAPO-style clip-higher, dynamic sampling that drops groups whose rollouts all score the same, token-level loss, and overlong-response shaping. Dr.GRPO-style removal of length and std normalization.
   - **Sequence-level importance ratio (GSPO)**, needed once the student is a mixture-of-experts model.
   - **Truncated or masked importance sampling** to correct for the mismatch between the vLLM sampler and the trainer.
   - **Rank-GRPO** (ICLR 2026) for recommendation lists. Each rank position is its own action with its own reward, which trains better than one sequence-level NDCG score.
   - **Rewards by task:**
     - recommendations: verifiable rewards (NDCG and Recall against held-out purchases, hard constraints, product ID in the candidate set, valid JSON);
     - copy: attribute-faithfulness checks, rubric scores from GenRM, and a language-ID gate;
     - image briefs: schema validity plus a VLM product-fidelity score on the rendered image.
   - **Guards against reward hacking:** KL anchor to the reference model, length control, a judge ensemble, a canary set scored by a *different* judge than the one used for training, and weekly human audits of the highest-reward samples.
4. **On-policy distillation from the RL'd 27B into the fast reranker.**
   - The fast model samples; the 27B scores every token by reverse KL. This is dense and much more step-efficient than GRPO.
   - Start with a short forward-KL warmup, then pure OPD.
   - Optionally, finish with a short round of student RL once the student has caught up.
5. **On-policy self-distillation (optional, cheap boost).**
   - The *same* model, given privileged context (the ground-truth next purchase, the full attribute sheet, the rubric), acts as the teacher for its own unprivileged rollouts.
   - This gives a dense signal without another model (following ROSD and Self-Distilled Policy Gradient, 2026).
6. **Continual loop.** Every N weeks, new crawl data and production feedback go through stages 2→3→4, gated by the frozen evals (M5).

**Images (optional track).**
- A LoRA on Qwen-Image-2.1 for product fidelity.
- Then preference optimization for the diffusion model (Diffusion-DPO or Flow-GRPO), using human image preferences plus the VLM product-fidelity reward.

**Training platform (all behind `train/base.py`)**
- **Primary: Tinker (Thinking Machines).**
  - You write the training loop in Python; Tinker runs it on its distributed GPUs with LoRA, from 1B up to 1T+ parameters, dense or mixture-of-experts, text and vision.
  - The `tinker-cookbook` has recipes that map directly onto the stages above: chat SFT, DPO and three-stage RLHF (SFT → reward model → RL), RL with verifiable rewards, on-policy and off-policy distillation (single or multiple teachers), and RL with tool use.
  - Custom losses allow the DAPO/GSPO/Rank-GRPO advantage functions to be written directly.
  - This removes GPU-cluster work from the project. Modal hosts the parts Tinker doesn't: rollout environments (candidate retrieval, graders, the judge, image rendering for rewards), data jobs and serving.
  - Trained checkpoints are downloaded as archives (`get_checkpoint_archive_url_from_tinker_path`) to a Modal Volume and served with vLLM.
- **First task in M3:** list the models this Tinker account can train (the cookbook's server capabilities call) and confirm Qwen3.8-27B is there.
  - If it isn't yet, use Qwen3.5-27B (documented as supported) or Inkling-Small on Tinker. The Modal fallback below can also train Qwen3.8-27B directly.
- **Fallback, self-run on Modal:** TRL + PEFT (SFT, DPO, GenRM) and verl (GRPO/DAPO/GSPO, Rank-GRPO, OPD, async vLLM rollouts, LoRA adapter-only sync) on H100/H200 clusters. Use it when a model or loss isn't available on Tinker.
- **Storage and tracking:** checkpoints on a Modal Volume, W&B for runs, and every run pinned to a data, grader and benchmark version.
- **Scale-up path:** iterate first on a small proxy (about 4B) with the full recipe, then scale to 27B.
- **AC2 later:** add `train/ac2/` (following the cookbook's `tau2bench` / `dapo-math-check` pattern) and switch by config. No task or grader code changes.

**Export.** HF-format weights and LoRA adapters are saved to a Modal Volume or a private HF repo, together with the config, data version, grader version and eval report.

**Technique watch.** `docs/POST_TRAINING_NOTES.md` is reviewed every quarter. Any new method is added as a new `Trainer` or advantage-function plug-in and must beat the current recipe on the frozen evals before it replaces it.

## 4. Benchmarks and evaluation
Benchmarks come in two layers:
- **public benchmarks**, for comparing with other published work and catching regressions;
- **C2A-Bench**, a new benchmark built for this project, because none of the public ones measure grounded recommendations and copy on *our* catalogs across SE, UK, ES, US and CA.

All of them run in one harness (`src/c2a/eval/`, built on Inspect AI plus custom scorers) with the `c2a bench` CLI.

**4a. Public benchmarks to adopt** (licenses to be checked for each before use)
| Area | Benchmark | What it measures for us |
|---|---|---|
| Search relevance | **Amazon ESCI / SQID** (130K queries, 2.6M labels in Exact/Substitute/Complement/Irrelevant; en/es/ja; SQID adds images) | Query→product relevance, and substitute-vs-complement judgement (the core of cross-sell) |
| Session recommendations and text | **Amazon-M2** (KDD Cup 2023; multilingual, includes UK and ES locales) | Next-item prediction and product title generation across locales |
| Shopping knowledge | **Shopping MMLU** (KDD Cup 2024) and **ECInstruct** (10 tasks, 264K examples, including sequential recommendation and query-product ranking) | Product-concept understanding; guards against regressions from fine-tuning |
| Shopping agents | **ShoppingBench** (AAAI), **ShoppingComp**, **Shopping Companion** (2026), **ComboShoppingBench** (budget and coupons, 2026), **WebMall** / **ShopGym** | Intent-grounded, multi-step and constrained shopping, preference grounding, and safety-critical product choices |
| Customer-service style tool use | **τ-bench retail** | Post-checkout flows (returns, exchanges, order questions) |
| General capability guard | A small fixed suite (instruction following, multilingual, safety) | Catches general regressions caused by domain RL |

**4b. C2A-Bench (built by this project, versioned and frozen per release)**
- **Construction:**
  - generated from our crawl plus the open behavioural datasets;
  - held out **by store and by time** (products and stores never seen in training);
  - split by locale (sv-SE, en-GB, es-ES, en-US, en-CA, fr-CA) and by vertical.
- **Tracks:**
  1. **Rec-Pre:** search, browse and "similar to this product" reranking over 50 candidates. Metrics: NDCG@10, Recall@10, constraint violations, hallucinated-ID rate.
  2. **Rec-Post:** cross-sell and replenishment. Metrics: complement precision, substitute-confusion rate, co-purchase Recall@k.
  3. **Copy:** titles, descriptions, bullets, SEO text and ads. Metrics: attribute-faithfulness (every claim checked against attributes), locale/language accuracy, rubric score, human preference win-rate.
  4. **Localize:** translate and adapt copy between locales while preserving attributes, units, sizes and currency.
  5. **Image:** briefs rendered by the image model. Metrics: VLM product-fidelity score, text/logo accuracy, human preference.
  6. **Conversational / agentic:** multi-turn shopping assistant in a simulated store over our catalog (UCP-style search and cart tools), scored on task success, constraint adherence and turns taken.
  7. **Decide:** calibrated decisions (relevance, complement vs substitute, claim faithfulness, intent, constraints). Metrics: weighted accuracy, ECE, Brier score, plus router cost and latency savings.
- **Gold subset:** about 500 items per track and locale, verified by native-speaker annotators through the labelling app. It is used to calibrate the automatic judges (report judge–human agreement), and the headline numbers come from it.
- **Hygiene:**
  - rules against data contamination (dedup C2A-Bench against every training source using MinHash);
  - a canary string in the benchmark files so any leak into training data can be detected;
  - a private holdout split that is never used for tuning;
  - eval judges that differ from the reward judges.
- **Leaderboard:** each run's report (parquet + HTML) is compared against all baselines: base candidates, teacher, SFT, RL, OPD. It includes per-locale and per-vertical breakdowns plus cost and latency.

**4c. Gates**
- **M2 bake-off:** pick the student (Qwen3.8-27B vs Inkling-Small), the teacher, the image model and the decider (Decision-1.0-Lux-9B as shipped vs fine-tuned vs distilled) on C2A-Bench (gold subset) plus the public benchmarks.
- **Ship gate:**
  - the student beats its base model on every C2A-Bench track;
  - it reaches ≥90% of teacher quality on the Rec and Copy tracks;
  - hallucinated-ID rate is under 0.5%;
  - no public benchmark drops by more than 2 points.
- **Online:** counterfactual (IPS/DR) estimates on logged traffic, then an A/B test.

## 5. Serving on Modal
- `serve/llm.py`: vLLM with Qwen3.8-27B plus task LoRAs (`--enable-lora`), OpenAI-compatible, autoscaling with warm pool, structured JSON output (guided decoding).
- `serve/image.py`: Qwen-Image-2.1 on L40S/H100, generation plus editing using the crawled reference images.
- `serve/gateway.py`: FastAPI. It retrieves candidates from LanceDB, calls the LLM, validates output against the pydantic schema, runs the image brief through the image model, and logs requests and feedback (clicks, add-to-cart) to parquet for the next training round (continual learning on AC2).

## Milestones
1. **M0 skeleton (done):** repo layout, schemas, CLI, compliance, Shopify parser, graders, decide interface, training data formats, rewards and advantage estimators, Tinker SFT/RL/OPD loops, gateway, Modal stubs, CI (ruff + pytest).
2. **M1 data:** registry and discovery, Shopify/UCP/Firecrawl ingestion for about 10 stores, open-dataset loaders, normalization, index.
3. **M2 datasets, graders and benchmarks:** task builders, teacher distillation with a cost cap, public benchmark adapters, C2A-Bench v0 with a gold subset, and the **model bake-off** (student, teacher and image model).
4. **M3 post-training on Tinker:** cold-start SFT → GenRM + rubrics → Rank-GRPO/DAPO RL → OPD into the fast reranker, on the proxy model first and then on 27B. (Optional M3b: port to AC2 once access arrives.)
   - **M3a** runs alongside M3 and collects human preferences through the labelling app.
5. **M4 serving:** vLLM + image + gateway on Modal, load test.
6. **M5 feedback loop:** logging → dataset refresh → retrain schedule.

## Verification
- Unit tests: schema round-trip, Shopify/UCP parsers against recorded fixtures, grader correctness on hand-built cases (including hallucinated-ID and unfaithful-claim negatives), split leakage test.
- Smoke E2E (`c2a e2e --tiny`): 2 stores → 200 products → 100 examples per task → 20-step LoRA SFT on Modal → eval → deploy to a dev Modal endpoint → curl `/recommend`, `/generate/text` and `/generate/image` and validate the responses against the schemas.
- Full run: eval report comparing base, SFT, RL and teacher, plus a gateway latency/throughput test.

## Open items to confirm during implementation
- Exact HF model IDs and licenses for Qwen3.8-27B, Inkling-Small, Qwen-Image-2.1 and FLUX.2, plus Kimi K3 / GLM-5.3 API access and pricing.
- Which models this Tinker account can train (Qwen3.8-27B?), Tinker pricing, and the checkpoint export format for vLLM.
- Licenses of the public benchmarks for commercial use.
- Decision-1.0 model license, input/output format and training recipe (the HF card and vllm-sr.ai blog were not reachable from this environment).
- AC2 SDK/config specifics (the docs were not reachable from this environment; the cookbook structure is used as the template).
- Legal sign-off on each tier-B/C store before it is enabled in `registry.yaml`.
