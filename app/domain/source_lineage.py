"""Program-verified source inventory, kept separate from quote/fact evidence.

No model, repository, workflow, network or gold is loaded by this module.
The caller supplies immutable paragraph rows and source-verified candidates.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import re
from urllib.parse import urlsplit
from uuid import NAMESPACE_URL, uuid5

VERSION = 'known-source-lineage-v1'
COUNT_BASIS = ('program_inventory_of_immutable_documents_in_cited_known_origin_components; '
               'fully_operator_excluded_documents_not_active; '
               'different_components_are_not_proven_independent; not_source_quote_facts')


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _digest(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def _text(value):
    if not isinstance(value, str) or not value:
        raise ValueError('invalid_lineage_identity')
    return value


def _known_groups(paragraphs):
    """Same conservative exact-identity rule as the existing workflow.

    It detects known links, never the independence of different components.
    No model-provided source_group_id is used to establish a source link.
    """
    parents = {p['document_id']: p['document_id'] for p in paragraphs}
    def root(doc):
        while parents[doc] != doc:
            parents[doc] = parents[parents[doc]]
            doc = parents[doc]
        return doc
    signals, document_signals = {}, {d: set() for d in parents}
    for p in paragraphs:
        doc = p['document_id']
        for field, prefix in (('content_sha256', 'content'), ('content_url', 'explicit_url')):
            if not p.get(field):
                continue
            signal = prefix + ':' + p[field]
            document_signals[doc].add(signal)
            if signal in signals:
                left, right = sorted((root(doc), root(signals[signal])))
                parents[right] = left
            else:
                signals[signal] = doc
    groups = defaultdict(set)
    for doc in parents:
        groups[root(doc)].update(document_signals[doc])
    return {doc: (
        str(uuid5(NAMESPACE_URL, 'trust-judge:source:' + min(groups[root(doc)] or {'document:' + root(doc)}))),
        'known_exact_hash_or_explicit_url_only' if groups[root(doc)] else 'same_document_only_not_complete_origin_detection',
    ) for doc in parents}


def _safe_url(value):
    """Expose only a URL-shaped data value, never credentials or free prose."""
    if not value:
        return None, 'not_supplied'
    try:
        parsed = urlsplit(value)
        safe = (parsed.scheme in {'http', 'https'} and parsed.hostname and
                parsed.username is None and parsed.password is None and
                len(value) <= 4096 and not re.search(r'\s', value))
    except ValueError:
        safe = False
    return (value, 'stored_explicit_metadata_not_web_verified') if safe else (None, 'unsafe_or_non_url_metadata_not_exposed')


def _overlap(left, right):
    return left[0] < right[1] and right[0] < left[1]


def _fully_covered(length, spans):
    end = 0
    for start, stop in sorted(spans):
        if start > end:
            return False
        end = max(end, stop)
    return end >= length


def _prepare(selected, candidates, paragraphs, excluded_spans):
    if not all(isinstance(value, list) for value in (selected, candidates, paragraphs)):
        raise ValueError('invalid_lineage_collections')
    para_map, docs, raw_paragraphs = {}, {}, set()
    for raw in paragraphs:
        if not isinstance(raw, dict):
            raise ValueError('invalid_lineage_paragraph')
        p = {key: raw.get(key) for key in ('id', 'document_id', 'raw_doc_id', 'raw_para_id', 'text', 'content_sha256', 'content_url')}
        for field in ('id', 'document_id', 'raw_doc_id', 'raw_para_id'):
            _text(p[field])
        if p['id'] in para_map or not isinstance(p['text'], str) or not p['text']:
            raise ValueError('invalid_lineage_paragraph')
        if (p['document_id'], p['raw_para_id']) in raw_paragraphs:
            raise ValueError('duplicate_lineage_raw_paragraph_id')
        raw_paragraphs.add((p['document_id'], p['raw_para_id']))
        for field in ('content_sha256', 'content_url'):
            if p[field] is not None and not isinstance(p[field], str):
                raise ValueError('invalid_lineage_document_metadata')
        if p['content_sha256'] and not re.fullmatch('[0-9a-f]{64}', p['content_sha256']):
            raise ValueError('invalid_lineage_document_hash')
        identity = {key: p[key] for key in ('document_id', 'raw_doc_id', 'content_sha256', 'content_url')}
        if p['document_id'] in docs and docs[p['document_id']]['identity'] != identity:
            raise ValueError('inconsistent_lineage_document_metadata')
        docs.setdefault(p['document_id'], {'identity': identity, 'paragraphs': []})['paragraphs'].append(p)
        para_map[p['id']] = p
    if len({d['identity']['raw_doc_id'] for d in docs.values()}) != len(docs):
        raise ValueError('duplicate_lineage_raw_document_id')
    groups = _known_groups(list(para_map.values()))
    excluded = defaultdict(list)
    for raw in excluded_spans:
        if not isinstance(raw, (tuple, list)) or len(raw) != 4:
            raise ValueError('invalid_lineage_exclusion')
        pid, start, end, quote = raw
        p = para_map.get(pid)
        if (p is None or type(start) is not int or type(end) is not int or
                not 0 <= start < end <= len(p['text']) or p['text'][start:end] != quote):
            raise ValueError('invalid_lineage_exclusion')
        if (start, end) not in excluded[pid]:
            excluded[pid].append((start, end))
    def verify(row):
        p = para_map.get(row.get('paragraph_id'))
        start, end = row.get('start_cp'), row.get('end_cp')
        if (p is None or type(start) is not int or type(end) is not int or
                not 0 <= start < end <= len(p['text']) or p['text'][start:end] != row.get('quote')):
            raise ValueError('invalid_lineage_candidate_span')
        group, basis = groups[p['document_id']]
        if ((row.get('raw_doc_id'), row.get('raw_para_id')) != (p['raw_doc_id'], p['raw_para_id']) or
                row.get('source_group_id') != group or row.get('source_group_basis') != basis):
            raise ValueError('invalid_lineage_candidate_provenance')
        _text(row.get('id'))
        return p, any(_overlap((start, end), span) for span in excluded[p['id']])
    pool = {}
    for row in candidates:
        if not isinstance(row, dict) or row.get('id') in pool:
            raise ValueError('duplicate_or_invalid_lineage_candidate')
        p, blocked = verify(row)
        pool[row['id']] = (row, p, blocked)
    seen, selected_groups = set(), {}
    for row in selected:
        if not isinstance(row, dict) or row.get('id') in seen:
            raise ValueError('duplicate_or_invalid_selected_lineage')
        seen.add(row['id'])
        p, blocked = verify(row)
        if blocked:
            raise ValueError('selected_lineage_overlaps_operator_exclusion')
        anchor_id = row.get('source_evidence_id') or row['id']
        if anchor_id not in pool:
            # A preserved human fact-row may be the only supplied candidate
            # for its immutable geometric source. Permit its exact self-entry,
            # never an invented or model-owned source anchor.
            own = pool.get(row['id'])
            geometric_id = str(uuid5(NAMESPACE_URL, 'trust-judge:evidence:' + json.dumps(
                [row['paragraph_id'], row['start_cp'], row['end_cp'], row['quote']],
                ensure_ascii=False, sort_keys=True, separators=(',', ':'))))
            if (row.get('interpretation_origin') != 'human' or own is None
                    or own[0].get('interpretation_origin') != 'human'
                    or own[0].get('source_evidence_id') != anchor_id or anchor_id != geometric_id
                    or any(own[0][key] != row[key] for key in ('paragraph_id', 'start_cp', 'end_cp', 'quote'))):
                raise ValueError('selected_lineage_anchor_missing_from_candidates')
            anchor, original_p, _ = own
        else:
            anchor, original_p, _ = pool[anchor_id]
        if (original_p['id'] != p['id'] or not anchor['start_cp'] <= row['start_cp'] < row['end_cp'] <= anchor['end_cp']):
            raise ValueError('selected_lineage_outside_original_anchor')
        selected_groups[row['id']] = groups[p['document_id']][0]
    return para_map, docs, groups, excluded, pool, selected_groups


def _catalogs(selected, candidates, paragraphs, excluded_spans):
    para_map, docs, groups, excluded, pool, selected_groups = _prepare(selected, candidates, paragraphs, excluded_spans)
    catalogs = {}
    for group_id in sorted(set(selected_groups.values())):
        members = []
        for did in sorted(docs, key=lambda key: (docs[key]['identity']['raw_doc_id'], key)):
            if groups[did][0] != group_id:
                continue
            d = docs[did]
            full_excluded = all(_fully_covered(len(p['text']), excluded[p['id']]) for p in d['paragraphs'])
            touched = any(excluded[p['id']] for p in d['paragraphs'])
            url, url_status = _safe_url(d['identity']['content_url'])
            windows = []
            for cid, (row, p, blocked) in sorted(pool.items()):
                if p['document_id'] != did:
                    continue
                windows.append({'candidate_evidence_id': cid, 'paragraph_id': p['id'], 'raw_para_id': p['raw_para_id'],
                    'start_cp': row['start_cp'], 'end_cp': row['end_cp'], 'quote_sha256': _sha(row['quote']),
                    'status': 'operator_excluded_overlap_metadata_only' if blocked else 'verified_candidate_window_metadata_only'})
            exclusion_audit = [{'paragraph_id': p['id'], 'raw_para_id': p['raw_para_id'], 'start_cp': start, 'end_cp': end,
                'quote_sha256': _sha(p['text'][start:end]), 'reason': 'operator_excluded_span_no_quote_or_fact_disclosed'}
                for p in sorted(d['paragraphs'], key=lambda p: p['id']) for start, end in sorted(excluded[p['id']])]
            # Fully excluded documents retain only identity/hash/exclusion audit;
            # source prose, quote/fact text, credentials and instruction text never appear.
            member = {'document_id': did, 'raw_doc_id': d['identity']['raw_doc_id'],
                'document_content_sha256': d['identity']['content_sha256'],
                'source_group_basis': groups[did][1], 'source_paragraph_count': len(d['paragraphs']),
                'exclusion_status': 'fully_operator_excluded' if full_excluded else 'partially_operator_excluded' if touched else 'not_operator_excluded',
                'candidate_windows': windows, 'excluded_span_audit': exclusion_audit,
                'source_url': None if full_excluded else url,
                'source_url_status': 'not_disclosed_fully_operator_excluded' if full_excluded else url_status,
                'source_url_sha256': _sha(d['identity']['content_url']) if d['identity']['content_url'] else None}
            members.append(member)
        counts = {'active_source_documents': sum(m['exclusion_status'] != 'fully_operator_excluded' for m in members),
                  'known_origin_groups': int(any(m['exclusion_status'] != 'fully_operator_excluded' for m in members)),
                  'basis': COUNT_BASIS}
        content = {'schema_version': VERSION, 'kind': 'program_metadata_not_source_quote_or_fact',
            'source_group_id': group_id, 'source_group_members': members, 'source_counts': counts,
            'independent_source_count': None,
            'interpretation': 'Members of this component are one known linked source group, not separate independent votes. Different components are not proven independent. Metadata proves supplied source presence and linkage only, not event truth or source authority. Member windows are not additional citable evidence IDs.'}
        catalogs[group_id] = {'catalog_id': _digest(content), **content}
    return catalogs, selected_groups


def attach_source_group_members(selected_evidence, canonical_candidates, paragraphs, *, excluded_spans=()):
    """Deep-copy selected rows and attach one shared catalog per selected group.

    Refined rows may bind their original canonical anchor with source_evidence_id;
    their narrower quote must remain inside it and pass exact source validation.
    Existing/model-supplied lineage/counts are overwritten, never trusted.
    """
    catalogs, selected_groups = _catalogs(selected_evidence, canonical_candidates, paragraphs, excluded_spans)
    owners = {group: min(eid for eid, gid in selected_groups.items() if gid == group) for group in catalogs}
    output = deepcopy(selected_evidence)
    for row in output:
        group = selected_groups[row['id']]
        catalog = catalogs[group]
        if row['id'] == owners[group]:
            row['source_lineage'] = {**deepcopy(catalog), 'catalog_owner_evidence_id': owners[group], 'inline_members': True}
        else:
            row['source_lineage'] = {'schema_version': VERSION, 'kind': 'program_metadata_not_source_quote_or_fact',
                'source_group_id': group, 'catalog_id': catalog['catalog_id'],
                'catalog_owner_evidence_id': owners[group], 'inline_members': False,
                'source_counts': deepcopy(catalog['source_counts']), 'independent_source_count': None}
        row.pop('source_group_members', None)
        row['source_counts'] = deepcopy(catalog['source_counts'])
    return output


def get_program_source_counts(evidence, candidate_evidence, paragraphs, *, excluded_spans=()):
    """Recompute restricted counts from immutable inputs for exactly cited rows.

    This never reads counts, IDs' digits, dates, URL digits or quote text as a
    numeric whitelist. Counts are not evidence of any unrelated event number.
    """
    catalogs, _ = _catalogs(evidence, candidate_evidence, paragraphs, excluded_spans)
    return {'active_source_documents': sum(c['source_counts']['active_source_documents'] for c in catalogs.values()),
            'known_origin_groups': sum(c['source_counts']['known_origin_groups'] for c in catalogs.values()),
            'basis': COUNT_BASIS}
