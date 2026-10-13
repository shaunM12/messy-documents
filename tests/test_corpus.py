import json

from docsearch.config import PROJECT_ROOT

MANIFEST = PROJECT_ROOT / "data" / "manifest.json"


def test_manifest_files_exist_and_cover_all_types():
    entries = json.loads(MANIFEST.read_text())
    suffixes = set()
    for entry in entries:
        assert (PROJECT_ROOT / entry["path"]).is_file(), entry["path"]
        assert entry["visibility"]
        suffixes.add(entry["path"].rsplit(".", 1)[-1])
    assert {"md", "csv", "pdf"} <= suffixes


def test_handbook_versions_differ_on_changed_facts():
    v1 = (PROJECT_ROOT / "data/raw/policy_handbook_v1.md").read_text()
    v2 = (PROJECT_ROOT / "data/raw/policy_handbook_v2.md").read_text()
    assert "within 30 days of purchase" in v1
    assert "within 14 days of purchase" in v2
