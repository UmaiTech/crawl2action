"""`c2a` command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from c2a import NotYetImplemented
from c2a.config import load_settings

app = typer.Typer(
    help="crawl2action: crawl -> datasets -> post-train -> bench -> serve", no_args_is_help=True
)
registry_app = typer.Typer(help="Store registry", no_args_is_help=True)
train_app = typer.Typer(help="Post-training (Tinker primary; Modal fallback)", no_args_is_help=True)
app.add_typer(registry_app, name="registry")
app.add_typer(train_app, name="train")
label_app = typer.Typer(
    help="Label scraped data with hosted Jev (System One)", no_args_is_help=True
)
app.add_typer(label_app, name="label")


def _pending(what: str, milestone: str) -> None:
    typer.secho(str(NotYetImplemented(what, milestone)), fg=typer.colors.YELLOW, err=True)
    raise typer.Exit(code=2)


@registry_app.command("validate")
def registry_validate(path: Path | None = typer.Option(None, help="registry.yaml path")) -> None:
    """Validate the store registry and summarize it."""
    from c2a.sources.registry import load_registry

    reg = load_registry(path)
    by_country: dict[str, int] = {}
    for s in reg.stores:
        by_country[s.country] = by_country.get(s.country, 0) + 1
    typer.echo(
        f"{len(reg.stores)} stores: " + ", ".join(f"{c}={n}" for c, n in sorted(by_country.items()))
    )
    typer.echo(f"crawlable (enabled + ToS approved): {len(reg.crawlable())}")


@app.command()
def config(path: Path | None = typer.Option(None)) -> None:
    """Print resolved settings."""
    typer.echo(load_settings(path).model_dump_json(indent=2))


BackendOpt = typer.Option(
    "keyword", help="keyword (offline) | jev (hosted, needs TYPESAFE_API_KEY)"
)


def _make_decider(backend: str):
    from c2a.decide.backends import KeywordDecider, SystemOneClient

    if backend == "keyword":
        return KeywordDecider()
    if backend == "jev":
        lab = load_settings().labeler
        return SystemOneClient(base_url=lab.base_url, model=lab.model)
    raise typer.BadParameter(f"unknown backend: {backend}")


@app.command()
def decide(
    evidence: str = typer.Option(...),
    question: str = typer.Option("Which label applies?"),
    labels: str = typer.Option(..., help="comma-separated"),
    backend: str = BackendOpt,
) -> None:
    """Run a single decision (evidence + question + runtime labels -> probabilities)."""
    from c2a.decide.base import decide as run_decide

    labels_list = [x.strip() for x in labels.split(",") if x.strip()]
    result = run_decide(_make_decider(backend), evidence, question, labels_list)
    typer.echo(result.model_dump_json(indent=2))


ConfigOpt = typer.Option(..., "--config", exists=True, dir_okay=False)
DryRunOpt = typer.Option(False, "--dry-run", help="validate config + dataset only")


@train_app.command("sft")
def train_sft(config_path: Path = ConfigOpt, dry_run: bool = DryRunOpt) -> None:
    """Supervised fine-tuning (cold start)."""
    _train("sft", config_path, dry_run)


@train_app.command("rl")
def train_rl(config_path: Path = ConfigOpt, dry_run: bool = DryRunOpt) -> None:
    """RL with grader rewards (GRPO / Dr.GRPO / Rank-GRPO)."""
    _train("rl", config_path, dry_run)


@train_app.command("opd")
def train_opd(config_path: Path = ConfigOpt, dry_run: bool = DryRunOpt) -> None:
    """On-policy distillation from a same-tokenizer teacher."""
    _train("opd", config_path, dry_run)


def _train(stage: str, config_path: Path, dry_run: bool) -> None:
    from c2a.train.base import load_run_config
    from c2a.train.data import DistillPrompt, RLPrompt, SFTExample, read_jsonl

    cfg = load_run_config(config_path)
    if cfg.stage != stage:
        raise typer.BadParameter(f"config stage is '{cfg.stage}', not '{stage}'")
    record = {"sft": SFTExample, "rl": RLPrompt, "opd": DistillPrompt}[stage]
    n = None
    if cfg.dataset.exists():
        n = sum(1 for _ in read_jsonl(cfg.dataset, record))
    typer.echo(json.dumps({"run": cfg.model_dump(mode="json"), "dataset_records": n}, indent=2))
    if dry_run:
        if n is None:
            typer.secho(f"dataset not found: {cfg.dataset} (built in M2)", fg=typer.colors.YELLOW)
        return
    if n is None:
        raise typer.BadParameter(f"dataset not found: {cfg.dataset}")
    if cfg.platform != "tinker":
        _pending(f"platform '{cfg.platform}'", "M3")

    from c2a.train.tinker.client import TinkerSDKBackend

    backend = TinkerSDKBackend(cfg.base_model, cfg.lora_rank, cfg.teacher_model, cfg.max_length)
    records = list(read_jsonl(cfg.dataset, record))
    if cfg.stage == "sft":
        from c2a.train.tinker.sft import run_sft as run
    elif cfg.stage == "rl":
        from c2a.train.tinker.rl import run_rl as run
    else:
        from c2a.train.tinker.opd import run_opd as run
    summary = run(backend, records, cfg)
    typer.echo(f"done: {summary['steps']} steps, final checkpoint {summary['checkpoints'][-1]}")


def _load_label_items(input_path: Path, target: str):
    from c2a.labeling.records import LabelItem
    from c2a.labeling.state import product_state
    from c2a.schemas import Product
    from c2a.train.data import read_jsonl

    if target == "product":
        return [LabelItem(id=p.id, state=product_state(p)) for p in read_jsonl(input_path, Product)]
    return list(read_jsonl(input_path, LabelItem))


@label_app.command("run")
def label_run(
    input_path: Path = typer.Option(..., "--input", exists=True, dir_okay=False),
    qset_name: str = typer.Option("product_v1", "--qset"),
    out: Path = typer.Option(Path("data/labels"), "--out"),
    backend: str = typer.Option("jev", help="jev (hosted) | keyword (offline)"),
    max_requests: int | None = typer.Option(None, "--max-requests"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print a sample request, no API call"),
) -> None:
    """Label items (products JSONL for product_v1; LabelItem JSONL otherwise)."""
    from c2a.decide.systemone import SystemOneRequest
    from c2a.labeling.cache import ResponseCache
    from c2a.labeling.escalation import append_review_queue
    from c2a.labeling.gating import load_thresholds
    from c2a.labeling.pipeline import label
    from c2a.labeling.questions import get_question_set
    from c2a.train.data import write_jsonl

    qset = get_question_set(qset_name)
    items = _load_label_items(input_path, qset.target)
    if not items:
        raise typer.BadParameter("no items in input")
    model = load_settings().labeler.model
    if dry_run:
        sample = SystemOneRequest(state=items[0].state, model=model, questions=qset.questions)
        typer.echo(f"{len(items)} items, question set {qset.id}; sample request:")
        typer.echo(sample.model_dump_json(indent=2, exclude_none=True))
        return
    run_dir = out / qset.id.replace("@", "_v")
    result = label(
        items,
        _make_decider(backend),
        qset,
        load_thresholds(),
        cache=ResponseCache(out / "cache"),
        max_requests=max_requests,
        model=model,
    )
    write_jsonl(run_dir / "items.jsonl", items)
    write_jsonl(run_dir / "labels.jsonl", result.records)
    append_review_queue(run_dir / "review_queue.jsonl", result.review)
    accepted = sum(r.status == "accepted" for r in result.records)
    typer.echo(
        f"{len(result.records)} labels ({accepted} accepted, {len(result.review)} to review); "
        f"{result.requests_made} requests, {result.cache_hits} cache hits, "
        f"{len(result.skipped)} items skipped (budget) -> {run_dir}"
    )


@label_app.command("export")
def label_export(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    qset_name: str = typer.Option("product_v1", "--qset"),
    kind: str = typer.Option("decider-train", help="decider-train | gold"),
    out: Path = typer.Option(..., "--out"),
    include_jev_only: bool = typer.Option(
        False, help="include Jev-only labels (only if TypeSafe's terms allow training on them)"
    ),
) -> None:
    """Export accepted labels as decider training JSONL (Kev format) or C2A-Bench gold."""
    from c2a.labeling.export import DEFAULT_TRAIN_SOURCES, to_decider_train, to_gold
    from c2a.labeling.questions import get_question_set
    from c2a.labeling.records import LabelItem, LabelRecord
    from c2a.train.data import read_jsonl

    qset = get_question_set(qset_name)
    records = list(read_jsonl(run_dir / "labels.jsonl", LabelRecord))
    if kind == "gold":
        rows = to_gold(records)
    elif kind == "decider-train":
        states = {it.id: it.state for it in read_jsonl(run_dir / "items.jsonl", LabelItem)}
        sources = set(DEFAULT_TRAIN_SOURCES) | ({"jev"} if include_jev_only else set())
        rows = to_decider_train(states, records, qset, sources)
    else:
        raise typer.BadParameter(f"unknown kind: {kind}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    typer.echo(f"wrote {len(rows)} rows -> {out}")


@app.command()
def discover() -> None:
    """Discover stores (Firecrawl search + platform/UCP fingerprinting)."""
    _pending("store discovery", "M1")


@app.command()
def crawl() -> None:
    """Crawl enabled, ToS-approved stores."""
    _pending("crawling", "M1")


@app.command()
def build() -> None:
    """Build task datasets and C2A-Bench."""
    _pending("dataset build", "M2")


@app.command()
def bench() -> None:
    """Run public benchmarks + C2A-Bench."""
    _pending("benchmark harness", "M2")


@app.command()
def deploy() -> None:
    """Deploy serving apps to Modal."""
    _pending("Modal deployment", "M4")


if __name__ == "__main__":
    app()
