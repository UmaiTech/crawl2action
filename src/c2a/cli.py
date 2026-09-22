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


@app.command()
def decide(
    evidence: str = typer.Option(...),
    question: str = typer.Option("Which label applies?"),
    labels: str = typer.Option(..., help="comma-separated"),
    endpoint: str | None = typer.Option(
        None, help="OpenAI-compatible decider URL; default: keyword backend"
    ),
    model: str | None = typer.Option(None),
) -> None:
    """Run a single decision (evidence + question + runtime labels -> probabilities)."""
    from c2a.decide.backends import KeywordDecider, OpenAICompatDecider
    from c2a.schemas import DecisionRequest

    req = DecisionRequest(
        evidence=evidence,
        question=question,
        labels=[x.strip() for x in labels.split(",") if x.strip()],
    )
    decider = (
        OpenAICompatDecider(endpoint, model or load_settings().models.decider)
        if endpoint
        else KeywordDecider()
    )
    typer.echo(decider.decide(req).model_dump_json(indent=2))


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
