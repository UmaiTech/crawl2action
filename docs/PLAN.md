# crawl2action: plan for an end-to-end pipeline from e-commerce crawling to a post-trained LLM

## Context
The `crawl2action` repo is empty apart from a README and LICENSE. The goal is one pipeline that:
1. crawls product catalogs from popular e-commerce sites in SE, UK, ES, US and CA using **Firecrawl**, plus **UCP catalog endpoints** and open datasets;
2. turns that data into training and eval sets for **pre-checkout recommendations** (search/browse/PDP), **post-checkout recommendations** (cross-sell, replenishment), **text generation** (product descriptions, titles, ads, localized copy) and **image generation** (product and lifestyle visuals);
3. post-trains a **student LLM** on **Modal** with the current recipe (cold-start SFT → reward modelling with human preferences, rubrics and GenRM → GRPO-family RL → on-policy distillation) for now, with **Applied Compute AC2** as a later drop-in behind the same trainer interface;
4. serves the models on **Modal**.

Decisions confirmed with the user:
- **Two model tiers:**
  - The student is **Qwen 3.6-27B** (dense). It is the model that gets post-trained and served in production on 1–2 H100/H200.
  - The teacher and judge is **Kimi K3** (2.8T MoE, 104B active) or Qwen 3.8-Max, called through a hosted API. It is used only for generating synthetic data and for grading, never for serving.
  - Qwen3.8-Flash-Next (multimodal MoE) is an optional later upgrade if image input for the student proves useful.
- **Images:** the LLM writes structured image briefs, and a separate image model renders them. That model is **Qwen-Image-2.1** (7B, generation plus editing, 10 reference images), served on Modal. An optional LoRA can be trained on crawled product photos.
- **Data sources:** a mix of popular regional stores, open datasets and UCP catalog access. Every source goes through robots.txt and ToS gating.

## Architecture (six stages, each a Modal app plus a `c2a` CLI command)
```
discover → crawl/ingest → normalize+dedupe → build datasets → train (Modal-TRL; AC2 later) → eval → serve (Modal)
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
  train/
    base.py             # Trainer interface: submit(dataset, config) -> run_id; export(run_id) -> HF/LoRA weights
    sft.py              # TRL cold-start SFT / DPO baselines (Modal)
    reward/             # preference-labelling app, rubric generation, GenRM training
    rl/                 # verl on Modal: GRPO/DAPO/GSPO losses, Rank-GRPO advantage, TIS correction
    opd.py              # on-policy distillation + self-distillation
    ac2/                # LATER: AC2 project (added once access is granted)
  eval/                 # offline eval harness → reports (parquet + HTML summary)
  serve/
    llm.py              # Modal + vLLM, OpenAI-compatible, --enable-lora for per-task adapters
    image.py            # Modal + diffusers Qwen-Image-2.1 (+ optional product LoRA)
    gateway.py          # FastAPI on Modal: /recommend, /cross-sell, /generate/text, /generate/image, /feedback
  cli.py                # c2a discover|crawl|build|train|eval|deploy
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

## 3. Post-training (state of the art as of Sep 2026)
The design follows the **sparse-to-dense reward principle** (Microsoft Research, May 2026):
- Spend **sparse, outcome-level reward** (RL) on the model best able to explore with it.
- Pass the resulting behaviour to the deployed model through **dense, token-level supervision**, meaning on-policy distillation (OPD).
- Run student-side RL only after that distillation step.

Qwen3, GLM-5 and MiMo all use OPD in their post-training pipelines. Running plain GRPO directly on a cold student wastes the labelled signal on the least-prepared policy.

**Model roles**
| Role | Model | Why |
|---|---|---|
| Frontier teacher / data generator / rubric writer | Kimi K3 or Qwen 3.8-Max (hosted API) | Best quality. It is only used through text outputs, because its tokenizer differs from the student's, so it cannot provide token-level logprobs for OPD |
| **Main model (RL'd)** | Qwen 3.6-27B (self-hosted on Modal) | Serves copy, image briefs and conversational recommendations. It is also the same-tokenizer teacher for OPD into the fast model |
| Fast reranker (optional) | A small Qwen (about 4–9B, same tokenizer family) | High-QPS `/recommend`. Trained with OPD from the RL'd 27B model |
| Reward model / judge | A small Qwen trained as a generative reward model (GenRM) | Cheap, fast reward for RL. It is calibrated against Kimi K3 and human labels |

**Stages** (every stage is a separate, resumable Modal job that writes a versioned checkpoint and an eval report)
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

**Frameworks on Modal (all behind `train/base.py`):**
- **TRL + PEFT** for cold-start SFT, GenRM SFT and DPO-style preference baselines.
- **verl** for GRPO/DAPO/GSPO, Rank-GRPO (custom advantage function) and OPD. It has the most complete support for asynchronous rollouts with LoRA adapter-only weight sync, and uses vLLM for rollouts.
  - Switch to TRL's async trainer later if it reaches parity.
  - Reconsider slime or prime-rl if the student becomes a mixture-of-experts model.
- **GPUs:** a single H100 for proxy-scale runs; multi-GPU H100/H200 (Modal clustered functions) for 27B RL.
- **Storage and tracking:** checkpoints on a Modal Volume, W&B for runs, and every run pinned to a data and grader version.
- **Scale-up path:** iterate first on a small proxy (about 4B) with the full recipe, then scale to 27B.
- **AC2 later:** its console for inspecting rollouts, grader iteration and support for self-distillation line up with stages 3–5. Once access arrives, add `train/ac2/` (following the cookbook's `tau2bench` / `dapo-math-check` pattern) and switch by config. No task or grader code changes.

**Export.** HF-format weights and LoRA adapters are saved to a Modal Volume or a private HF repo, together with the config, data version, grader version and eval report.

**Technique watch.** `docs/POST_TRAINING_NOTES.md` is reviewed every quarter. Any new method is added as a new `Trainer` or advantage-function plug-in and must beat the current recipe on the frozen evals before it replaces it.

## 4. Evaluation
- The `c2a eval` CLI runs base Qwen 3.6-27B, the SFT checkpoint, the RL checkpoint, the OPD fast model and the teacher on the frozen eval sets.
- **Eval judges are separate from reward judges** (a different model plus a human sample), so RL cannot overfit to the scorer.
- **Recommendations** are also scored with counterfactual (IPS/DR) estimates on logged production traffic before any A/B test.
- Reports: per-task metrics, per-locale breakdown, cost/latency, and a small human spot-check sample.
- Ship gate: the student beats base and reaches ≥90% of teacher quality on the rec and copy tasks, with a hallucinated-ID rate under 0.5%.

## 5. Serving on Modal
- `serve/llm.py`: vLLM with Qwen 3.6-27B plus task LoRAs (`--enable-lora`), OpenAI-compatible, autoscaling with warm pool, structured JSON output (guided decoding).
- `serve/image.py`: Qwen-Image-2.1 on L40S/H100, generation plus editing using the crawled reference images.
- `serve/gateway.py`: FastAPI. It retrieves candidates from LanceDB, calls the LLM, validates output against the pydantic schema, runs the image brief through the image model, and logs requests and feedback (clicks, add-to-cart) to parquet for the next training round (continual learning on AC2).

## Milestones
1. **M0 skeleton:** repo layout, schemas, CLI, Modal app stubs, CI (ruff + pytest).
2. **M1 data:** registry and discovery, Shopify/UCP/Firecrawl ingestion for about 10 stores, open-dataset loaders, normalization, index.
3. **M2 datasets and graders:** all task builders, teacher distillation with a cost cap, frozen evals, baseline eval of base model vs teacher.
4. **M3 post-training on Modal:** cold-start SFT → GenRM + rubrics → Rank-GRPO/DAPO RL → OPD into the fast reranker, on the proxy model first and then on 27B. (Optional M3b: port to AC2 once access arrives.)
   - **M3a** runs alongside M3 and collects human preferences through the labelling app.
5. **M4 serving:** vLLM + image + gateway on Modal, load test.
6. **M5 feedback loop:** logging → dataset refresh → retrain schedule.

## Verification
- Unit tests: schema round-trip, Shopify/UCP parsers against recorded fixtures, grader correctness on hand-built cases (including hallucinated-ID and unfaithful-claim negatives), split leakage test.
- Smoke E2E (`c2a e2e --tiny`): 2 stores → 200 products → 100 examples per task → 20-step LoRA SFT on Modal → eval → deploy to a dev Modal endpoint → curl `/recommend`, `/generate/text` and `/generate/image` and validate the responses against the schemas.
- Full run: eval report comparing base, SFT, RL and teacher, plus a gateway latency/throughput test.

## Open items to confirm during implementation
- Exact HF model IDs and licenses for Qwen 3.6-27B, Qwen-Image-2.1 and Kimi K3 API access and pricing.
- AC2 SDK/config specifics (the docs were not reachable from this environment; the cookbook structure is used as the template).
- Legal sign-off on each tier-B/C store before it is enabled in `registry.yaml`.
