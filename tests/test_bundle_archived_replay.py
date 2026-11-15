"""Frozen application replay uses synthetic cases only; no model or network."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.archival

import app.domain.source_bundle_tools as live_tools
import app.workflows.bundle_workbench as workbench
import scripts.source_bundle_tool as live_entry
from test_bundle_workbench import Q, descriptor, response_fixture, sources


ROOT = Path(__file__).resolve().parents[1]


def full_response(tmp_path, *, changed="scripts/source_bundle_tool.py"):
    docs, rows = sources()
    response, _ = response_fixture(tmp_path, docs, rows)
    names = [p.relative_to(ROOT).as_posix() for p in (ROOT / "app").rglob("*.py")]
    names.append("scripts/source_bundle_tool.py")
    archives = response["proof"]["source_archives"] = {
        name: descriptor(tmp_path / "full-archive" / name, (ROOT / name).read_bytes(), raw=True) for name in names}
    if changed:
        old = Path(archives[changed]["file"]).read_bytes() + b"\n# Frozen older build with identical behavior.\n"
        archives[changed] = descriptor(tmp_path / "full-archive" / changed, old, raw=True)
    return docs, rows, response


def forbid_live_entry(*_args, **_kwargs):
    raise AssertionError("Changed archived codec must never use the live business entry")


def test_matching_minimal_codec_archive_keeps_fast_path_without_subprocess(tmp_path, monkeypatch):
    docs, rows = sources()
    response, _ = response_fixture(tmp_path, docs, rows)
    monkeypatch.setattr(workbench.subprocess, "run", forbid_live_entry)
    result = workbench.compile_result(Q, rows, docs, response)
    assert result["source_bundle_proof"]["native_source_delivery_replayed"]


@pytest.mark.parametrize("changed", ["scripts/source_bundle_tool.py", "app/domain/source_bundle_tools.py",
                                     "app/domain/query_context.py"])
def test_changed_entry_or_transitive_codec_runs_frozen_CLI_without_import_pollution(tmp_path, monkeypatch, changed):
    docs, rows, response = full_response(tmp_path, changed=changed)
    original_path, original_module = list(sys.path), sys.modules[live_tools.__name__]
    real_run, calls = subprocess.run, []
    # An unrelated archived script must not become a second executable entry.
    response["proof"]["source_archives"]["scripts/unapproved_entry.py"] = descriptor(
        tmp_path / "unapproved.py", b"raise RuntimeError('must never execute')", raw=True)

    def capture(args, **kwargs):
        archived_root = Path(kwargs["cwd"])
        assert args[:3] == [sys.executable, "-I", "-B"]
        assert Path(args[3]) == archived_root / "scripts/source_bundle_tool.py"
        assert Path(args[4]) == archived_root / "raw-case.json"
        assert not (archived_root / "scripts/unapproved_entry.py").exists()
        assert set(kwargs["env"]) <= {"SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
        calls.append(archived_root)
        return real_run(args, **kwargs)

    monkeypatch.setattr(live_entry, "execute", forbid_live_entry)
    monkeypatch.setattr(workbench.subprocess, "run", capture)
    result = workbench.compile_result(Q, rows, docs, response)
    assert result["evidence"][0]["quote"] == response["artifact"]["evidence"][0]["quote"]
    if changed == "scripts/source_bundle_tool.py":
        assert workbench.validate_result(result, Q, rows, docs) == result
    assert len(calls) == (4 if changed == "scripts/source_bundle_tool.py" else 2)
    assert all(not p.exists() for p in calls)
    assert sys.path == original_path and sys.modules[live_tools.__name__] is original_module


def test_changed_codec_with_only_four_files_cannot_claim_complete_archive(tmp_path):
    docs, rows = sources()
    response, _ = response_fixture(tmp_path, docs, rows)
    name = "app/domain/source_bundle_tools.py"
    response["proof"]["source_archives"][name] = descriptor(tmp_path / "changed.py", (ROOT / name).read_bytes() + b"\n# older\n", raw=True)
    with pytest.raises(ValueError, match="archived_application_required"):
        workbench.compile_result(Q, rows, docs, response)


def test_missing_transitive_module_fails_in_isolated_app_instead_of_live_fallback(tmp_path, monkeypatch):
    docs, rows, response = full_response(tmp_path)
    del response["proof"]["source_archives"]["app/domain/retrieval.py"]
    monkeypatch.setattr(live_entry, "execute", forbid_live_entry)
    with pytest.raises(ValueError, match="archived_replay_failed"):
        workbench.compile_result(Q, rows, docs, response)


def test_archived_behavior_is_compared_instead_of_silently_using_current_rules(tmp_path):
    docs, rows, response = full_response(tmp_path)
    name = "scripts/source_bundle_tool.py"
    data = (ROOT / name).read_text("utf-8")
    # This synthetic frozen revision deliberately differs from the recorded
    # response. The live entry would accept it; the real archived entry cannot.
    replacement = """
_original_execute = execute
def execute(*args, **kwargs):
    return {**_original_execute(*args, **kwargs), 'synthetic_old_codec_marker': True}
"""
    data = data.replace('if __name__ == "__main__":', replacement + '\nif __name__ == "__main__":')
    response["proof"]["source_archives"][name] = descriptor(tmp_path / "old-entry.py", data.encode("utf-8"), raw=True)
    with pytest.raises(ValueError, match="tool_transcript_replay_failed"):
        workbench.compile_result(Q, rows, docs, response)


@pytest.mark.parametrize("relative", ["app/../outside.py", "app//domain/duplicate.py", "C:/unapproved.py", "/absolute.py"])
def test_archive_paths_cannot_escape_execution_root(tmp_path, relative):
    docs, rows, response = full_response(tmp_path)
    response["proof"]["source_archives"][relative] = descriptor(tmp_path / "extra.py", b"pass\n", raw=True)
    with pytest.raises(ValueError, match="archive_relative_path_required"):
        workbench.compile_result(Q, rows, docs, response)


def test_normalized_duplicate_archive_path_is_rejected(tmp_path):
    docs, rows, response = full_response(tmp_path)
    response["proof"]["source_archives"]["app\\domain\\source_bundle_tools.py"] = deepcopy(
        response["proof"]["source_archives"]["app/domain/source_bundle_tools.py"])
    with pytest.raises(ValueError, match="archive_relative_path_required"):
        workbench.compile_result(Q, rows, docs, response)


def test_archive_tampering_is_rejected_before_subprocess(tmp_path, monkeypatch):
    docs, rows, response = full_response(tmp_path)
    Path(response["proof"]["source_archives"]["app/domain/retrieval.py"]["file"]).write_bytes(b"changed")
    monkeypatch.setattr(workbench.subprocess, "run", forbid_live_entry)
    with pytest.raises(ValueError, match="file_sha256_changed"):
        workbench.compile_result(Q, rows, docs, response)


def test_archived_replay_timeout_fails_closed_and_cleans_temporary_root(tmp_path, monkeypatch):
    docs, rows, response = full_response(tmp_path)
    roots = []
    def timeout(args, **kwargs):
        roots.append(Path(kwargs["cwd"]))
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])
    monkeypatch.setattr(workbench.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="archived_replay_unavailable"):
        workbench.compile_result(Q, rows, docs, response)
    assert roots and all(not root.exists() for root in roots)
