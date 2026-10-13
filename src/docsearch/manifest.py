import json
from pathlib import Path
from typing import Protocol

from docsearch.config import MANIFEST_PATH, PROJECT_ROOT


class Ingester(Protocol):
    def ingest(
        self, path: Path, *, doc_id: str, version: str, visibility: list[str], updated_at: str
    ): ...


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())


def ingest_manifest(target: Ingester, path: Path = MANIFEST_PATH) -> None:
    """Ingest every entry in order; later versions of a doc_id replace earlier ones."""
    for entry in load_manifest(path):
        target.ingest(
            PROJECT_ROOT / entry["path"],
            doc_id=entry["doc_id"],
            version=entry["version"],
            visibility=entry["visibility"],
            updated_at=entry["updated_at"],
        )
