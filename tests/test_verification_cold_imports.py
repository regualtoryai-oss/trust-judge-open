"""Import-order regressions run in genuinely fresh, network-disabled processes."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("first", ["strict_verification", "compiled_scope", "provider"])
def test_verification_modules_cold_import_in_any_supported_entry_order(first):
    modules = {"strict_verification": "app.domain.strict_verification",
               "compiled_scope": "app.domain.compiled_scope", "provider": "app.agents.provider"}
    order = [modules[first], *[value for key, value in modules.items() if key != first]]
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib,json,pathlib,socket,sys
def no_network(*args,**kwargs):
    raise AssertionError('cold_import_network_attempt')
socket.getaddrinfo=no_network
socket.socket.connect=no_network
loaded=[importlib.import_module(name) for name in json.loads(sys.argv[1])]
root=pathlib.Path.cwd().resolve()
assert all(pathlib.Path(module.__file__).resolve().is_relative_to(root/'app') for module in loaded)
from app.domain.compiled_scope import build_compiled_scope_request,resolve_compiled_scope
from app.domain.strict_verification import validate_scope_ledger
q={'question':'Synthetic cold import question?'}
payload,schema,views=build_compiled_scope_request(q,[])
ledger=resolve_compiled_scope({'candidate_catalog_id':payload['candidate_catalog_id'],'target_catalog_id':payload['target_catalog_id'],'decisions':{}},q,[])
assert validate_scope_ledger(ledger,q,[])==ledger and not views
print(json.dumps({'modules':[module.__name__ for module in loaded],'validated_empty_ledger':True}))
"""
    result = subprocess.run([sys.executable, "-B", "-c", script, json.dumps(order)], cwd=root,
                            text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"modules": order, "validated_empty_ledger": True}
