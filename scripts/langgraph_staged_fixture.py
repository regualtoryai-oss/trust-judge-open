"""Two fixed StateGraph nodes and MockTransport; no real model/network/credentials."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from typing import TypedDict
from unittest.mock import patch

import httpx
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

ROOT = Path(__file__).resolve().parents[1]
sha = lambda data: hashlib.sha256(data).hexdigest()


class State(TypedDict):
    messages: list
    stage: str


def _run(output: Path, manifest_path: Path) -> dict:
    output = output.resolve()
    assert output.is_relative_to((ROOT / '.runtime').resolve())
    output.mkdir()  # Exclusive fixture only.
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    raw_path = Path(manifest['casePath']); raw_bytes = raw_path.read_bytes()
    assert sha(raw_bytes) == manifest['caseSha256']
    raw = json.loads(raw_bytes)
    plan_path = Path(manifest['fixturePlan'])
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    dsh_dir = Path(manifest['outputDirectory'])
    dsh_wire_path = dsh_dir / 'fixture-wire.jsonl'
    dsh_wires = [json.loads(row) for row in dsh_wire_path.read_text(encoding='utf-8').splitlines()]
    receipt_path = dsh_dir / 'closed-receipt.json'
    dsh_receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert len(dsh_wires) == 2 and dsh_receipt['fixture'] is True and dsh_receipt['closed'] is True
    dsh_profile_path = dsh_dir / 'profile.patch.yml'
    profile = json.loads(dsh_profile_path.read_text(encoding='utf-8'))
    budget_config = next(row['config'] for patch_row in profile for row in patch_row.get('insert', [])
                         if row['id'] == 'trust-judge-episode-budget')
    shared_sources = [ROOT / 'integrations/dsh' / name for name in
        ('staged_evidence.mjs', 'evidence_tools.mjs', 'evidence_policy.mjs', 'budget_counter.mjs', 'budget_core.mjs')]
    for source in shared_sources:
        binding = next(row for row in budget_config['sourceBindings'] if Path(row['file']).resolve() == source.resolve())
        assert sha(source.read_bytes()) == binding['sha256']
    env = {key: os.environ[key] for key in ('SystemRoot', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP', 'PATH', 'PATHEXT') if key in os.environ}
    bridge_path = ROOT / 'scripts/langgraph_fixture_bridge.mjs'
    artifact_path = output / 'business-artifact.json'
    bridge = subprocess.Popen(['node', str(bridge_path), str(manifest_path), str(artifact_path)],
        cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8')
    def command(value):
        assert bridge.stdin is not None and bridge.stdout is not None
        bridge.stdin.write(json.dumps(value, ensure_ascii=False) + '\n'); bridge.stdin.flush()
        line = bridge.stdout.readline()
        if not line:
            raise RuntimeError('fixture_bridge_closed')
        response = json.loads(line)
        if not response['ok']:
            raise RuntimeError(response['code'])
        return response['result']
    wires, native_counts, raw_wire_bindings, call_index, expected_visible_tools = [], [], [], 0, []
    def transport(request: httpx.Request):
        nonlocal call_index
        assert request.url.host == 'fixture.invalid' and request.method == 'POST'
        body_bytes = request.content
        body = json.loads(body_bytes)
        assert call_index < 2 and body['model'] == 'deepseek-v4-pro'
        expected = plan[call_index]
        assert [t['function']['name'] for t in body['tools']] == [expected['name']]
        assert body['tools'] == expected_visible_tools
        counted = command({'op': 'count', 'serialized_body': body_bytes.decode('utf-8')})
        assert counted['wire_sha256'] == sha(body_bytes)
        raw_wire_path = output / f'native-openai-request-{call_index + 1}.json'
        with raw_wire_path.open('xb') as handle:
            handle.write(body_bytes)
        raw_wire_bindings.append({'file': str(raw_wire_path), 'sha256': sha(body_bytes),
                                  'input_tokens': counted['input_tokens']})
        native_counts.append(counted)
        wires.append(body)
        response = {'id': f'fixture-{call_index}', 'object': 'chat.completion', 'created': 0, 'model': 'deepseek-v4-pro',
          'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': f'fixture-tool-{call_index}', 'type': 'function', 'function': {
              'name': expected['name'], 'arguments': json.dumps(expected['arguments'], ensure_ascii=False)}}]}, 'finish_reason': 'tool_calls'}],
          'usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}}
        call_index += 1
        return httpx.Response(200, json=response)
    client = httpx.Client(transport=httpx.MockTransport(transport), trust_env=False)
    llm = ChatOpenAI(model='deepseek-v4-pro', api_key='fixture-only-not-a-real-credential',
        base_url='https://fixture.invalid/v1', http_client=client, max_retries=0, max_tokens=6144,
        extra_body={'thinking': {'type': 'disabled'}}, http_socket_options=())
    tool_results, system_equal = [], []
    def node(state: State, step: int):
        nonlocal expected_visible_tools
        visible = command({'op': 'state'})
        assert visible['system'] == dsh_wires[step]['system']
        system_equal.append(True)
        expected_visible_tools = visible['tools']
        response = llm.bind_tools(visible['tools']).invoke([SystemMessage(visible['system']), *state['messages']])
        assert len(response.tool_calls) == 1
        tool_call = response.tool_calls[0]
        expected = plan[step]
        assert tool_call['name'] == expected['name'] and tool_call['args'] == expected['arguments']
        result = command({'op': 'execute', 'name': tool_call['name'], 'arguments': tool_call['args']})
        tool_results.append(result['value'])
        assert result['ended'] is (step == 1)
        messages = [*state['messages'], response, ToolMessage(content=json.dumps(result['value'], ensure_ascii=False, separators=(',', ':')),
                    tool_call_id=tool_call['id'], name=tool_call['name'])]
        return {'messages': messages, 'stage': 'read' if step == 0 else 'submitted'}
    builder = StateGraph(State)
    builder.add_node('read', lambda state: node(state, 0))
    builder.add_node('submit', lambda state: node(state, 1))
    builder.add_edge(START, 'read'); builder.add_edge('read', 'submit'); builder.add_edge('submit', END)
    user_blocks = dsh_wires[0]['messages'][0]['content']
    try:
        final = builder.compile().invoke({'messages': [HumanMessage(content=user_blocks)], 'stage': 'select'})
        assert final['stage'] == 'submitted' and call_index == 2
        complete = tool_results[0]
        assert len(complete['paragraphs']) == 6
        tool_messages = [json.loads(m['content']) for m in wires[1]['messages'] if m['role'] == 'tool']
        assert len(tool_messages) == 1 and tool_messages[0]['paragraphs'] == complete['paragraphs']
        for ref, title, text in complete['paragraphs']:
            source = next(p for p in raw['catalog'] if p['paragraph_ref'] == ref)
            assert source['title'] == title and source['text'] == text
            assert sum(row[0] == ref for message in tool_messages for row in message['paragraphs']) == 1
        assert '"candidate_previews":' not in wires[1]['messages'][0]['content']
        lg_artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
        dsh_artifact_path = dsh_dir / 'submission.json'
        dsh_artifact = json.loads(dsh_artifact_path.read_text(encoding='utf-8'))
        canonical = lambda a: {k: a[k] for k in ('question_id', 'evidence', 'conclusion')}
        assert canonical(lg_artifact) == canonical(dsh_artifact)
        wire_path = output / 'native-openai-wires.jsonl'
        wire_path.write_bytes(('\n'.join(json.dumps(w, ensure_ascii=False, separators=(',', ':')) for w in wires) + '\n').encode())
        result = {'format': 'langgraph-two-node-staged-fixture-v1', 'closed': True, 'fixture': True, 'real_api_requests': 0,
          'mock_requests': 2, 'model_requests': 0, 'quality_metrics': None, 'fixture_usage_tokens_not_real_bill': 300,
          'framework': 'LangGraph StateGraph', 'version': importlib.metadata.version('langgraph'),
          'langchain_openai_version': importlib.metadata.version('langchain-openai'),
          'model_visible_system_strings_equal': system_equal, 'original_question_equal': True, 'candidate_count': 40,
          'full_source_count': 6, 'full_sources_equal': True, 'canonical_business_artifact_equal': True,
          'native_tools_equal_shared_business_definitions': True,
          'shared_business_sources_match_closed_dsh_profile': True,
          'native_protocol': 'OpenAI Chat Completions', 'dsh_native_protocol': 'Messages',
          'protocol_difference_prevents_pure_framework_causal_claim': True,
          'native_input_token_estimates': [r['input_tokens'] for r in native_counts],
          'original_native_wire_bindings': raw_wire_bindings,
          'native_input_token_estimate_basis': 'official_tokenizer_native_request_estimate',
          'output_reservation_each_request': 6144, 'estimate_not_billing_guarantee': True,
          'mock_latency_is_not_model_speed': True, 'fixed_graph_call_count_is_not_general_framework_advantage': True,
          'python_socket_connections_denied': True,
          'fixture_read_admission_total_zero_is_not_real_budget_guard': True,
          'real_paired_budget_admission_implemented': False, 'real_paired_run_authorized': False,
          'semantic_verified': False, 'publishable': False,
          'source_bindings': [{'file': str(p), 'sha256': sha(p.read_bytes())} for p in
            (Path(__file__), bridge_path, manifest_path, raw_path, plan_path, dsh_wire_path, receipt_path,
             dsh_profile_path, wire_path, artifact_path, *shared_sources,
             *(Path(r['file']) for r in raw_wire_bindings))]}
        (output / 'closed-receipt.json').write_bytes((json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
        return result
    finally:
        if bridge.stdin: bridge.stdin.close()
        bridge.wait(timeout=10)
        client.close()


def run(output: Path, manifest_path: Path) -> dict:
    def deny_network(*_args, **_kwargs):
        raise RuntimeError('fixture_network_forbidden')
    with patch('socket.socket.connect', deny_network), patch('socket.socket.connect_ex', deny_network), patch.dict(os.environ, {
        'LANGCHAIN_TRACING_V2': 'false', 'LANGSMITH_TRACING': 'false'}):
        return _run(output, manifest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.manifest), ensure_ascii=False))
