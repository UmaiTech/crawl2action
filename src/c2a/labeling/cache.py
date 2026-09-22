"""On-disk System One response cache so re-runs never pay for the same request twice."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from c2a.decide.systemone import SystemOneRequest, SystemOneResponse


def request_key(request: SystemOneRequest, model: str) -> str:
    body = request.model_dump(mode="json", exclude_none=True)
    body["model"] = model
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class ResponseCache:
    def __init__(self, directory: Path | str) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> SystemOneResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        return SystemOneResponse.model_validate_json(path.read_text())

    def put(self, key: str, response: SystemOneResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_dump_json())
        tmp.replace(path)
