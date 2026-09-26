import pytest
from pydantic import ValidationError

from c2a.train.data import Message, RLPrompt, SFTExample, read_jsonl, write_jsonl


def test_sft_must_end_with_assistant():
    with pytest.raises(ValidationError):
        SFTExample(
            id="1", messages=[Message(role="user", content="hi"), Message(role="user", content="?")]
        )


def test_jsonl_roundtrip(tmp_path):
    recs = [
        RLPrompt(id=str(i), messages=[Message(role="user", content="q")], reward="rec")
        for i in range(3)
    ]
    path = tmp_path / "x" / "rl.jsonl"
    assert write_jsonl(path, recs) == 3
    assert list(read_jsonl(path, RLPrompt)) == recs


def test_read_jsonl_reports_line(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"id": "1"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        list(read_jsonl(p, RLPrompt))
