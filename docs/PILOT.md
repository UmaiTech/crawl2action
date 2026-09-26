# Pilot runbook: scrape and label real stores

The goal of this pilot is to crawl 3–5 Shopify stores and label their products and product pairs with Jev. It also produces the first human-reviewed gold labels. Everything runs on your machine. This takes about 30 minutes plus review time.

## 1. Setup
```bash
git clone git@github.com:umaitech/crawl2action.git && cd crawl2action
git checkout claude/magical-davinci-7u43rl
uv sync --extra dev
cp .env.example .env        # then fill in the keys below
```
`.env` needs:
- `TYPESAFE_API_KEY`: required (Jev labeling).
- `FIRECRAWL_API_KEY`: only needed if you add a non-Shopify store.

The `.env` file is gitignored and is loaded automatically by every `c2a` command.

## 2. Pick and approve stores
Suggested pilot stores. They are believed to run on Shopify, which means no Firecrawl credits are needed for products; `doctor` and `fingerprint` will confirm this:

| id | store | country |
|---|---|---|
| `allbirds` | Allbirds | US |
| `kotn` | Kotn | CA |
| `tentree` | Tentree | CA |
| `djerfavenue` | Djerf Avenue | SE |
| `gymshark` | Gymshark | UK |

Review each store's terms of service and robots.txt yourself. Approve only stores whose terms allow automated collection of public product data for your purpose. Approval is recorded with your name:
```bash
uv run c2a registry approve allbirds --by "<your name>" --note "ToS reviewed <date>: ..."
uv run c2a discover fingerprint --store allbirds --write   # sets platform, method, currency
```
Commit `src/c2a/sources/registry.yaml` afterwards, so the approval and its audit trail are shared.

## 3. Preflight
```bash
uv run c2a doctor --live
```
Every line should show ✅. `--live` makes one real Jev call to confirm the key works and that our parser matches Jev's actual response format. If you see "Jev response format ❌", send the output to Claude: it includes the raw response body.

## 4. Run the pilot
```bash
uv run c2a pilot --store allbirds --store kotn --store tentree --max-products 100 --max-jev-requests 400
```
The pilot runs these steps:
1. It crawls each store politely: 1 request per second, following robots.txt and retrying on 429s.
2. It labels new or changed products with `product_v1`. That is one Jev request per product.
3. It builds candidate pairs inside each store (3 similar products plus 1 from another category, per anchor) and labels them with `pair_v1`.
4. It writes `data/reports/pilot.md`.

Re-running only pays for new or changed products and uncached requests.

**Cost check:** Jev requests ≈ products + pairs. With 3 stores × 100 products and `--pairs-k 3`, that is about 300 + 1,200. The run stops at `--max-jev-requests`, so start small.

## 5. Review uncertain labels
```bash
uv run c2a label review --run-dir data/labels/product_v1 --limit 30
uv run c2a label review --run-dir data/labels/pair_v1 --limit 30
```
For each item, pick a number, `s` to skip or `q` to quit. Every answer is saved immediately.

## 6. Look at the results
```bash
uv run c2a report
uv run c2a label export --run-dir data/labels/product_v1 --kind gold --out data/gold/product_v1.jsonl
uv run c2a label export --run-dir data/labels/pair_v1 --qset pair_v1 --kind gold --out data/gold/pair_v1.jsonl
```

## Send back to Claude
- the output of `c2a doctor --live`;
- `data/reports/pilot.md`;
- any `data/raw/<store>/manifest.json` with errors.

These say whether the real APIs and stores behave as the code expects, and how the confidence thresholds in `configs/labeling.yaml` should be tuned.
