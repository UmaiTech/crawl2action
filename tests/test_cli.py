from typer.testing import CliRunner

from c2a.bench import CANARY, TRACKS, contains_canary
from c2a.cli import app

runner = CliRunner()


def test_registry_validate():
    res = runner.invoke(app, ["registry", "validate"])
    assert res.exit_code == 0 and "crawlable (enabled + ToS approved): 0" in res.output


def test_decide_keyword_backend():
    res = runner.invoke(
        app,
        ["decide", "--evidence", "running socks for shoes", "--labels", "complement,substitute"],
    )
    assert res.exit_code == 0 and '"probs"' in res.output


def test_train_dry_run_and_stage_mismatch():
    res = runner.invoke(
        app, ["train", "sft", "--config", "configs/train/sft_student.yaml", "--dry-run"]
    )
    assert res.exit_code == 0 and "sft-student-proxy" in res.output
    res = runner.invoke(
        app, ["train", "rl", "--config", "configs/train/sft_student.yaml", "--dry-run"]
    )
    assert res.exit_code != 0


def test_stubs_exit_nonzero_with_milestone():
    res = runner.invoke(app, ["crawl"])
    assert res.exit_code == 2 and "M1" in res.output


def test_bench_tracks_and_canary():
    assert len(TRACKS) == 7 and contains_canary(f"x {CANARY} y")
