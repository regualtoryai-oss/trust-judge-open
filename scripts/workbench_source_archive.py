"""Read-only loading of SHA-bound archived codecs, with scoped import identity.

Only explicitly enumerated application codec dependencies are substituted, and
all substitutions are restored when the offline audit context exits. This has
no provider, repository, credential or reference access.
"""
from contextvars import ContextVar
from hashlib import sha256
import importlib.util
from pathlib import Path, PurePosixPath
import sys

ACTIVE = ContextVar("offline_workbench_source_archive", default=None)
_ABSENT = object()
DEPENDENCIES = {
    "app/agents/strict_evidence.py": ("app/domain/numeric_literals.py",),
    "app/agents/compact_selection.py": ("app/agents/strict_evidence.py",),
    "app/agents/scoped_arbitration.py": ("app/agents/schemas.py",),
    "app/domain/question_recall.py": ("app/domain/retrieval.py",),
    "app/workflows/strict_workbench.py": ("app/agents/strict_evidence.py", "app/domain/evidence_balance.py",
        "app/domain/material_eligibility.py", "app/domain/paragraph_ranked.py", "app/domain/retrieval.py",
        "app/domain/source_metadata.py", "app/domain/question_recall.py"),
}


def relative_source(value):
    if not isinstance(value, str):
        raise ValueError("only_explicit_app_python_source_archive_allowed")
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (not normalized.startswith("app/") or path.suffix != ".py" or path.is_absolute()
            or any(part in {"", ".", ".."} for part in normalized.split("/"))):
        raise ValueError("only_explicit_app_python_source_archive_allowed")
    return normalized


class SourceArchive:
    def __init__(self, descriptors):
        self.descriptors = {relative_source(key): value for key, value in descriptors.items()}
        if len(self.descriptors) != len(descriptors):
            raise ValueError("duplicate_normalized_source_archive_path")
        self.modules, self.saved, self.parent_saved = {}, {}, {}

    def __enter__(self):
        self.token = ACTIVE.set(self)
        return self

    def __exit__(self, *_error):
        for (parent, attribute), value in reversed(list(self.parent_saved.items())):
            module = sys.modules.get(parent)
            if module is not None:
                if value is _ABSENT:
                    if hasattr(module, attribute): delattr(module, attribute)
                else:
                    setattr(module, attribute, value)
        for name, value in reversed(list(self.saved.items())):
            if value is _ABSENT: sys.modules.pop(name, None)
            else: sys.modules[name] = value
        ACTIVE.reset(self.token)

    def load(self, relative, expected_sha256):
        relative = relative_source(relative)
        binding = self.descriptors.get(relative)
        if binding is None or binding.get("sha256") != expected_sha256:
            raise ValueError("archive_codec_not_bound_to_native_source_sha")
        path = Path(binding["file"])
        if sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("archived_source_bytes_changed")
        if relative in self.modules:
            return self.modules[relative]
        dependencies = list(DEPENDENCIES.get(relative, ()))
        # The old codec did not depend on provisional-role quarantine. Bind
        # the helper only when these exact archived source bytes import it.
        if (relative == "app/domain/analysis_wire_projection.py"
                and b"from app.domain.provisional_relations import" in path.read_bytes()):
            dependencies.append("app/domain/provisional_relations.py")
        for dependency in dependencies:
            if dependency not in self.descriptors:
                raise ValueError("archived_codec_dependency_missing")
            self.load(dependency, self.descriptors[dependency]["sha256"])
        name = relative.removesuffix(".py").replace("/", ".")
        parent, _, attribute = name.rpartition(".")
        if parent not in sys.modules:
            __import__(parent)
        self.saved.setdefault(name, sys.modules.get(name, _ABSENT))
        self.parent_saved.setdefault((parent, attribute), getattr(sys.modules[parent], attribute, _ABSENT))
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(sys.modules[parent], attribute, module)
        spec.loader.exec_module(module)
        self.modules[relative] = module
        return module


def active_archived_module(relative, expected_sha256):
    active = ACTIVE.get()
    return active.load(relative, expected_sha256) if active is not None and active.descriptors else None
