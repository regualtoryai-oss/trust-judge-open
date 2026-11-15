"""Reuse four pinned historical functions without its process-wide audit hook.

Only function definitions are compiled. Imports, assignments, hooks and CLI code
from the historical module are never executed. This helper is not a sandbox for
arbitrary source: the default historical bytes are pinned and returned as guards.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

MAIN=Path(__file__).resolve().parents[1]
HISTORICAL=MAIN/"scripts/reference_mapping.py"
HISTORICAL_SHA256="8d8729f6e928d37d8c6911f7e4a3045b81cd9fea9e8875b8edba1222622d704a"
FUNCTIONS=("key","question_union_reference","sha","evaluator")


def extract_reference_functions(source,expected_sha256,*,main=MAIN):
    """Extract exactly one undecorated, inert-definition node for each function."""
    path=Path(source).resolve();data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha256:
        raise ValueError("historical_reference_source_sha256_changed")
    tree=ast.parse(data.decode("utf-8-sig"),filename=str(path));selected=[]
    for name in FUNCTIONS:
        matches=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))and node.name==name]
        if len(matches)!=1 or not isinstance(matches[0],ast.FunctionDef):
            raise ValueError("one_synchronous_reference_function_required_"+name)
        node=matches[0];arguments=[*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs]
        if node.args.vararg:arguments.append(node.args.vararg)
        if node.args.kwarg:arguments.append(node.args.kwarg)
        if(node.decorator_list or node.args.defaults or any(v is not None for v in node.args.kw_defaults)
                or node.returns is not None or any(a.annotation is not None for a in arguments)
                or getattr(node,"type_params",[])):
            raise ValueError("reference_function_definition_must_be_inert_"+name)
        selected.append(node)
    namespace={"__name__":"offline_reference_functions_only","hashlib":hashlib,
        "importlib":importlib,"Path":Path,"MAIN":Path(main).resolve()}
    module=ast.Module(body=selected,type_ignores=[])
    exec(compile(module,str(path)+"::four_function_definitions","exec"),namespace)
    return SimpleNamespace(**{name:namespace[name]for name in FUNCTIONS})


def load_reference_tools():
    """Return (mapper, deterministic evaluator, historical/current source guards)."""
    mapper=extract_reference_functions(HISTORICAL,HISTORICAL_SHA256)
    engine,engine_guard=mapper.evaluator()
    guards=[{"file":str(HISTORICAL),"sha256":HISTORICAL_SHA256},engine_guard,
        {"file":str(Path(__file__).resolve()),"sha256":mapper.sha(__file__)}]
    return mapper,engine,guards
