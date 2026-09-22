# crawl2action — contributor notes

- Design: `docs/PLAN.md` (milestones M0–M5). Keep it in sync when decisions change.
- Setup: `uv sync --extra dev`. Checks (run before every push, same as CI):
  `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
- Optional integrations (tinker, modal, firecrawl, vllm) are extras. Import them lazily inside
  functions so the base install and tests never need them.
- Stubs raise `c2a.NotYetImplemented(what, milestone)`; CLI stubs exit with code 2.
- Crawling is default-deny: a store needs `enabled: true` AND `tos_status: approved`, and every
  URL must pass `compliance.can_crawl` + `RateLimiter`.
- Money is integer minor units + ISO currency.
- Graders are the single source of truth for quality: RL rewards (`train/rewards.py`), evals and
  serving guards all call them.
- Decisions use the System One contract (`decide/systemone.py`): hosted Jev labels data, and our
  post-trained decider serves the same API. Bump a question set's version whenever a question changes.
- Decider training exports leave out Jev-only labels unless TypeSafe's terms allow training on them.
- Never train on files containing `c2a.bench.CANARY`.
