# crawl2action

End-to-end pipeline for e-commerce AI:

**crawl** (Firecrawl, Shopify, UCP, open datasets) → **datasets** → **post-train** (Tinker; Modal fallback) → **benchmark** (public + C2A-Bench) → **serve** (Modal, behind vLLM Semantic Router).

It post-trains models for pre- and post-checkout recommendations, product copy, localization, image briefs and calibrated decisions. See [`docs/PLAN.md`](docs/PLAN.md) for the full design.

## Status: M0 (skeleton)
Working and tested:
- schemas (UCP-aligned)
- store registry (default-deny)
- crawl compliance (ToS + robots.txt + rate limit)
- Shopify `products.json` parser
- ranking metrics
- recommendation graders
- `decide` interface and calibration metrics
- training data formats
- RL rewards and advantage estimators (Dr.GRPO/GRPO, Rank-GRPO-style, DAPO dynamic sampling)
- Tinker SFT/RL/OPD loops (tested with a fake backend)
- the dev gateway

Everything else is a stub that names the milestone it lands in.

## Quickstart
```bash
uv sync --extra dev
uv run pytest -q
uv run c2a registry validate
uv run c2a decide --evidence "bought running shoes; candidate: running socks" \
  --labels exact,substitute,complement,irrelevant
uv run c2a train sft --config configs/train/sft_student.yaml --dry-run
```

To run real post-training on Tinker, run `uv sync --extra train`, set `TINKER_API_KEY`, add a dataset at the path in the run config, then:
```bash
uv run c2a train sft --config configs/train/sft_student.yaml
uv run c2a train rl  --config configs/train/rl_student_rank_grpo.yaml
uv run c2a train opd --config configs/train/opd_reranker.yaml
```

## Crawling
Crawling is default-deny: a store is crawled only after a person has reviewed its terms and approved it.
```bash
uv run c2a registry approve allbirds --by "<your name>" --note "ToS reviewed"
uv run c2a discover fingerprint --store allbirds --write   # ucp / shopify / firecrawl
uv run c2a crawl --dry-run                                  # what runs, and why others are skipped
export FIRECRAWL_API_KEY=... TYPESAFE_API_KEY=...
uv run c2a crawl --store allbirds --max-products 20 --label
uv run c2a discover expand --country SE --category apparel  # candidates for review
uv run c2a data import esci --path examples.parquet --path products.parquet --out data/open/esci
```
Parquet imports need `uv sync --extra data`.

## Labeling with Jev
Scraped products are classified and labeled by hosted **Jev** (TypeSafe System One) through `c2a label`. Uncertain labels are escalated to the teacher and then to human review.
```bash
export TYPESAFE_API_KEY=...
uv run c2a label run --input data/products.jsonl --qset product_v1 --max-requests 100
uv run c2a label export --run-dir data/labels/product_v1 --kind gold --out data/gold/product.jsonl
```
Add `--dry-run` to preview a request without calling the API, or `--backend keyword` to run offline.

## Layout
- `src/c2a/sources`: registry, compliance, Shopify/UCP/Firecrawl ingesters
- `src/c2a/graders`, `src/c2a/decide`: shared by evaluation, RL rewards and serving guards (`decide` uses the System One contract)
- `src/c2a/labeling`: Jev question sets, confidence gating, escalation, cache, exports
- `src/c2a/train`: data formats, rewards, `tinker/` loops, `fallback/` (Modal), `ac2/`
- `src/c2a/bench`, `src/c2a/eval`: C2A-Bench and the benchmark harness
- `src/c2a/serve`: gateway, vLLM and image Modal apps
- `configs/`: defaults, training run configs, semantic-router draft
