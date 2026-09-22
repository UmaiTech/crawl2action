# crawl2action: plan for an end-to-end pipeline from e-commerce crawling to a post-trained LLM

## Context
The `crawl2action` repo is empty apart from a README and LICENSE. The goal is one pipeline that:
1. crawls product catalogs from popular e-commerce sites in SE, UK, ES, US and CA using **Firecrawl**, plus **UCP catalog endpoints** and open datasets;
2. turns that data into training and eval sets for **pre-checkout recommendations** (search/browse/PDP), **post-checkout recommendations** (cross-sell, replenishment), **text generation** (product descriptions, titles, ads, localized copy) and **image generation** (product and lifestyle visuals);
3. post-trains a **student LLM** on **Modal** with TRL SFT + GRPO for now, with **Applied Compute AC2** as a later drop-in behind the same trainer interface;
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
    modal_trl.py        # PRIMARY: TRL SFT + GRPO on Modal H100/H200 (LoRA, vLLM rollouts)
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

## 3. Training
- **Stage 1 SFT (LoRA → optional full FT):** teacher-distilled, grader-filtered data. Uses a task mix config, trains 1–2 epochs, and checks formatting/JSON validity.
- **Stage 2 RL (GRPO-style) with graders as rewards.** The reward is a weighted blend of programmatic metrics and the LLM judge, with hard penalties for invalid JSON, hallucinated IDs and unfaithful claims.
- **Primary path, all on Modal (for now):** `train/modal_trl.py` runs SFT and GRPO with TRL + PEFT (LoRA) on Modal GPUs.
  - SFT runs on 1–2× H100.
  - GRPO for the 27B model runs on multi-GPU H100/H200, with vLLM generating rollouts (TRL `use_vllm`).
  - Checkpoints go to a Modal Volume, and runs are tracked in W&B or MLflow.
  - Iterate first on a small proxy student (the smallest current Qwen instruct model), then scale to 27B.
- **AC2 later:** the `Trainer` interface (`train/base.py`) and graders are platform-neutral. Once AC2 access arrives, add `train/ac2/` following the cookbook's `tau2bench` / `dapo-math-check` patterns and switch by config. No task or grader code changes.
- **Image LoRA (optional):** fine-tune Qwen-Image-2.1 on product photos for studio-style fidelity (on Modal).
- **Export:** HF-format weights / LoRA adapters, pushed to a Modal Volume or a private HF repo. Every run records its config, data version and eval report.

## 4. Evaluation
- The `c2a eval` CLI runs base Qwen 3.6-27B, the SFT checkpoint, the RL checkpoint and the teacher on the frozen eval sets.
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
4. **M3 training on Modal:** TRL SFT, then GRPO, on the proxy model first and then on 27B. (Optional M3b: port to AC2 once access arrives.)
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
