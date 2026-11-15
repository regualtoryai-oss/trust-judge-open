"""Synthetic mappings and loopback-only asyncio regression; no gold or APIs."""
import asyncio
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.offline_reference_tools import (
    FUNCTIONS,HISTORICAL,HISTORICAL_SHA256,extract_reference_functions,load_reference_tools,
)


SOURCE="""
raise RuntimeError('TOP_LEVEL_MUST_NOT_RUN')
def key(row): return row.get('doc_id'), row.get('para_id')
def question_union_reference(evidence, conclusion): return evidence, []
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def evaluator(): return MAIN
"""


def extract(tmp_path,source=SOURCE):
    path=tmp_path/"synthetic_legacy.py";path.write_text(source,encoding="utf-8")
    return extract_reference_functions(path,sha256(path.read_bytes()).hexdigest(),main=tmp_path)


def test_only_four_function_definitions_execute_not_module_imports_or_hooks(tmp_path):
    top="import sys\nsys.addaudithook(lambda event,args: (_ for _ in ()).throw(RuntimeError('HOOK_MUST_NOT_INSTALL')))\n"
    mapper=extract(tmp_path,top+SOURCE)
    assert set(vars(mapper))==set(FUNCTIONS)
    assert mapper.key({"doc_id":"d","para_id":"p"})==("d","p")
    assert mapper.evaluator()==tmp_path
    path=tmp_path/"data";path.write_bytes(b"synthetic")
    assert mapper.sha(path)==sha256(b"synthetic").hexdigest()


@pytest.mark.parametrize("variant",["missing","duplicate","decorator","async","default","annotation"])
def test_extraction_rejects_nonunique_or_active_definition_expressions(tmp_path,variant):
    source=SOURCE
    if variant=="missing":source=source.replace("def key(row): return row.get('doc_id'), row.get('para_id')\n","")
    elif variant=="duplicate":source+="\ndef key(row): return row\n"
    elif variant=="decorator":source=source.replace("def key(row)","@do_not_execute()\ndef key(row)")
    elif variant=="async":source=source.replace("def key(row)","async def key(row)")
    elif variant=="default":source=source.replace("def key(row)","def key(row=do_not_execute())")
    else:source=source.replace("def key(row)","def key(row: do_not_execute())")
    with pytest.raises(ValueError):extract(tmp_path,source)


def test_pinned_historical_sha_is_checked_before_definition_execution(tmp_path):
    path=tmp_path/"synthetic.py";path.write_text(SOURCE,encoding="utf-8")
    with pytest.raises(ValueError,match="sha256_changed"):
        extract_reference_functions(path,"0"*64)
    assert sha256(HISTORICAL.read_bytes()).hexdigest()==HISTORICAL_SHA256


def test_original_mapping_keeps_fact_identity_unresolved_rows_and_disagreement():
    mapper,engine,guards=load_reference_tools()
    evidence=[{"fact_id":"f1","doc_id":"d","para_id":"p1"},{"fact_id":"f2","doc_id":"d","para_id":"p1"},
        {"fact_id":"dup","doc_id":"d","para_id":"p2"},{"fact_id":"dup","doc_id":"d","para_id":"p3"},
        {"fact_id":"f3","doc_id":"d","para_id":"p5"}]
    conclusion={"supporting_facts":[{"fact_id":"f1"},{"fact_id":"f2"},{"doc_id":"d","para_id":"p4"},
        {"fact_id":"missing"},{"fact_id":"dup"},{"fact_id":"f3","doc_id":"different"}],"conflicting_facts":[{"fact_id":"f1"}]}
    rows,issues=mapper.question_union_reference(evidence,conclusion)
    assert len(rows)==6 and rows[:2]==[{"doc_id":"d","para_id":"p1"}]*2
    assert sum("_unresolved_reference"in row for row in rows)==2
    assert rows[-1]=={"doc_id":"d","para_id":"p5"}
    assert [r["check"]for r in issues].count("unknown_or_ambiguous_reference_fact_id_retained")==2
    assert [r["check"]for r in issues].count("reference_relation_disagrees_with_fact_source")==1
    metrics=engine.compare_rows("synthetic",[{"doc_id":"d","para_id":"p1"}],rows)["unique_paragraphs"]["A_id_only"]["metrics"]
    assert metrics["matches"]==1 and metrics["reference_records"]==5
    assert all(sha256(Path(g["file"]).read_bytes()).hexdigest()==g["sha256"]for g in guards)


def test_reference_tools_do_not_poison_a_new_event_loop_in_current_process():
    from scripts.score_obligation_selection_probe import reference_tools
    mapper,engine,guards=reference_tools()
    assert callable(mapper.key)and callable(engine.metrics)and len(guards)==4
    loop=asyncio.new_event_loop();loop.close()


def test_fresh_process_preserves_windows_asyncio_socketpair_after_scorer_loading():
    code="""
import asyncio,sys
from pathlib import Path
def guard(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes))and Path(str(args[0])).name.lower()=='.env':raise AssertionError('credentials forbidden')
 if event in {'socket.connect','socket.connect_ex','socket.bind'}and len(args)>1:
  address=args[1]
  if not isinstance(address,tuple)or address[0]not in {'127.0.0.1','::1','localhost'}:raise AssertionError('external network forbidden')
 if event=='socket.getaddrinfo'and args[0]not in {'127.0.0.1','::1','localhost',None}:raise AssertionError('external DNS forbidden')
sys.addaudithook(guard)
from scripts.score_obligation_selection_probe import reference_tools
for _ in range(2):
 mapper,engine,guards=reference_tools()
 assert mapper.key({'doc_id':'d','para_id':'p'})==('d','p')
 loop=asyncio.new_event_loop();loop.close()
"""
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,"-c",code],cwd=root,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
