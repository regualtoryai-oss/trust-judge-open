from hashlib import sha256
import sys

import pytest

from scripts.workbench_source_archive import SourceArchive, active_archived_module, relative_source


def test_archive_uses_original_bytes_and_restores_module_and_package_bindings(tmp_path):
    import app.agents
    name = "app.agents.example_archived_codec"
    before = sys.modules.get(name)
    path = tmp_path / "archived.py"
    path.write_bytes(b"def answer():\n    return 'old-frozen'\n")
    descriptor = {"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    with SourceArchive({"app/agents/example_archived_codec.py": descriptor}):
        module = active_archived_module("app/agents/example_archived_codec.py", descriptor["sha256"])
        assert module.answer() == "old-frozen"
        assert sys.modules[name] is module
        assert app.agents.example_archived_codec is module
    assert sys.modules.get(name) is before
    assert not hasattr(app.agents, "example_archived_codec")


def test_archive_hash_is_verified_before_executing_any_code(tmp_path):
    path = tmp_path / "changed.py"
    path.write_bytes(b"raise AssertionError('must not execute')")
    with SourceArchive({"app/agents/example_archived_codec.py": {"file": str(path), "sha256": "0" * 64}}):
        with pytest.raises(ValueError, match="archived_source_bytes_changed"):
            active_archived_module("app/agents/example_archived_codec.py", "0" * 64)


@pytest.mark.parametrize("path", ["../.env", "app/../.env", "scripts/arbitrary.py", "app/agents/../../outside.py", "/app/file.py",
    "app//agents/file.py", "app/./file.py", "app\\agents\\..\\file.py", None])
def test_source_archive_cannot_declare_credential_or_outside_module_paths(path):
    with pytest.raises(ValueError): relative_source(path)


def test_archive_rejects_duplicate_windows_and_posix_identity():
    with pytest.raises(ValueError, match="duplicate_normalized"):
        SourceArchive({"app/agents/example.py": {}, "app\\agents\\example.py": {}})


def test_missing_archived_codec_dependency_cannot_fall_back_to_mutable_live_module(tmp_path):
    path = tmp_path / "compact.py"
    path.write_bytes(b"raise AssertionError('must not execute without frozen dependency')")
    descriptor = {"file": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
    with SourceArchive({"app/agents/compact_selection.py": descriptor}):
        with pytest.raises(ValueError, match="archived_codec_dependency_missing"):
            active_archived_module("app/agents/compact_selection.py", descriptor["sha256"])
