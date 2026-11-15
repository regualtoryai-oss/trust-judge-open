"""Optional local dense recall: raw windows, immutable binding, parent-level RRF.

No provider/API/config import and no implicit model download. These scores are
retrieval hints; they never certify evidence, source truth or independence.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

VERSION = 'raw-window-semantic-v20'


class SemanticRecallUnavailable(RuntimeError):
    pass


def _canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')


def _numpy():
    try: import numpy
    except ImportError: raise SemanticRecallUnavailable('local_numpy_not_installed') from None
    return numpy


def source_manifest(paragraphs):
    if not isinstance(paragraphs,list): raise ValueError('semantic_sources_not_list')
    seen, locations, result = set(),set(),[]
    for row in paragraphs:
        if (not isinstance(row,dict) or any(not isinstance(row.get(key),str) or not row[key] for key in ('id','raw_doc_id','raw_para_id'))
                or not isinstance(row.get('text'),str)):
            raise ValueError('semantic_source_invalid')
        location=row['raw_doc_id'],row['raw_para_id']
        if row['id'] in seen or location in locations: raise ValueError('semantic_source_duplicate')
        seen.add(row['id']);locations.add(location)
        result.append({'id':row['id'],'raw_doc_id':row['raw_doc_id'],'raw_para_id':row['raw_para_id'],
                       'source_text_sha256':sha256(row['text'].encode('utf-8')).hexdigest()})
    return result


def window_catalog(paragraphs,*,window_chars=320,overlap_chars=80):
    if (type(window_chars)is not int or not 32<=window_chars<=1024 or type(overlap_chars)is not int
            or not 0<=overlap_chars<window_chars):raise ValueError('semantic_window_config_invalid')
    manifest=source_manifest(paragraphs);windows=[]
    for row,source in zip(paragraphs,manifest):
        text=row['text']
        if not text.strip():continue
        for start in range(0,len(text),window_chars-overlap_chars):
            end=min(len(text),start+window_chars);fragment=text[start:end]
            windows.append({'window_id':f'W{len(windows)+1}','paragraph_id':row['id'],
                'raw_doc_id':source['raw_doc_id'],'raw_para_id':source['raw_para_id'],
                'start_cp':start,'end_cp':end,'text':fragment,'source_text_sha256':source['source_text_sha256'],
                'window_text_sha256':sha256(fragment.encode('utf-8')).hexdigest()})
            if end==len(text):break
    return windows


def _normalized(values,*,expected_rows=None,dimensions=None):
    np=_numpy();matrix=np.asarray(values,dtype=np.float32)
    if matrix.ndim!=2 or not len(matrix) or matrix.shape[1]<1 or not np.isfinite(matrix).all():raise ValueError('semantic_vectors_invalid')
    if expected_rows is not None and len(matrix)!=expected_rows:raise ValueError('semantic_vector_row_count_mismatch')
    if dimensions is not None and matrix.shape[1]!=dimensions:raise ValueError('semantic_vector_dimension_mismatch')
    norms=np.linalg.norm(matrix,axis=1,keepdims=True)
    if np.any(norms<=0):raise ValueError('semantic_zero_vector')
    normalized=(matrix/norms).astype(np.float32,copy=True);normalized.setflags(write=False)
    return normalized


@dataclass(frozen=True)
class SemanticIndex:
    binding_sha256:str
    descriptor_json:str
    windows_json:str
    _vectors:object

    @property
    def descriptor(self):return json.loads(self.descriptor_json)


def build_index(paragraphs,windows,vectors,*,model_fingerprint,window_chars=320,overlap_chars=80):
    if not isinstance(model_fingerprint,str) or not model_fingerprint:raise ValueError('semantic_model_fingerprint_required')
    expected=window_catalog(paragraphs,window_chars=window_chars,overlap_chars=overlap_chars)
    if windows!=expected:raise ValueError('semantic_window_source_binding_mismatch')
    matrix=_normalized(vectors,expected_rows=len(windows))
    descriptor={'version':VERSION,'model_fingerprint':model_fingerprint,'source_manifest':source_manifest(paragraphs),
        'window_chars':window_chars,'overlap_chars':overlap_chars,'windows_sha256':sha256(_canonical(windows)).hexdigest(),
        'vectors_sha256':sha256(matrix.tobytes(order='C')).hexdigest(),'rows':len(windows),'dimensions':matrix.shape[1]}
    return SemanticIndex(sha256(_canonical(descriptor)).hexdigest(),_canonical(descriptor).decode(),_canonical(windows).decode(),matrix)


def rank_dense(index,query_vectors,*,limit=40,paragraphs=None):
    if not isinstance(index,SemanticIndex) or type(limit)is not int or not 1<=limit<=10000:raise ValueError('semantic_rank_parameters_invalid')
    np=_numpy();descriptor=index.descriptor;windows=json.loads(index.windows_json)
    if sha256(_canonical(descriptor)).hexdigest()!=index.binding_sha256:raise ValueError('semantic_index_descriptor_changed')
    if sha256(_canonical(windows)).hexdigest()!=descriptor['windows_sha256'] or sha256(index._vectors.tobytes(order='C')).hexdigest()!=descriptor['vectors_sha256']:
        raise ValueError('semantic_index_vectors_or_windows_changed')
    if paragraphs is not None and source_manifest(paragraphs)!=descriptor['source_manifest']:raise ValueError('semantic_index_source_changed')
    queries=_normalized(query_vectors,dimensions=descriptor['dimensions'])
    scores=np.max(queries@index._vectors.T,axis=0)
    byparent={};positions={row['id']:i for i,row in enumerate(descriptor['source_manifest'])}
    for window,score in zip(windows,scores):
        pid=window['paragraph_id'];score=float(score)
        if pid not in byparent or score>byparent[pid]['dense_cosine']:
            byparent[pid]={'id':pid,'raw_doc_id':window['raw_doc_id'],'raw_para_id':window['raw_para_id'],
                'source_text_sha256':window['source_text_sha256'],'dense_cosine':score,
                'best_window_id':window['window_id'],'best_window_text_sha256':window['window_text_sha256']}
    return sorted(byparent.values(),key=lambda row:(-row['dense_cosine'],positions[row['id']]))[:limit]


def fuse_candidates(paragraphs,ranked_lists,*,limit=40,rrf_k=60):
    manifest=source_manifest(paragraphs)
    if (type(limit)is not int or not 1<=limit<=10000 or type(rrf_k)is not int or rrf_k<1
            or not isinstance(ranked_lists,list) or not ranked_lists):raise ValueError('semantic_fusion_parameters_invalid')
    known={row['id']:row for row in manifest};positions={row['id']:i for i,row in enumerate(manifest)};scores={};signals={}
    for list_index,ranked in enumerate(ranked_lists):
        if not isinstance(ranked,list):raise ValueError('semantic_fusion_ranked_list_invalid')
        seen=set()
        for rank,row in enumerate(ranked,1):
            if not isinstance(row,dict) or row.get('id')not in known or row['id']in seen:raise ValueError('semantic_fusion_unknown_or_duplicate_id')
            pid=row['id'];seen.add(pid)
            for key in ('raw_doc_id','raw_para_id','source_text_sha256'):
                if key in row and row[key]!=known[pid][key]:raise ValueError('semantic_fusion_source_mismatch')
            scores[pid]=scores.get(pid,0)+1/(rrf_k+rank)
            signals.setdefault(pid,[]).append({'list':list_index,'rank':rank})
    return [{**known[pid],'fusion_score':scores[pid],'retrieval_signals':signals[pid]}
        for pid in sorted(scores,key=lambda pid:(-scores[pid],positions[pid]))[:limit]]


class FastEmbedLocal:
    """Explicit pre-downloaded E5-small ONNX only; never downloads in worker.

    The optional dependency belongs to a separately provisioned environment.
    Missing or changed local weights are unavailable, never an API fallback.
    """
    def __init__(self,folder,*,threads=4):
        path=Path(folder).resolve();manifest_path=path/'local-model-manifest.json'
        if not manifest_path.is_file():raise SemanticRecallUnavailable('local_embedding_model_not_prepared')
        manifest=json.loads(manifest_path.read_text('utf-8'))
        required={'onnx/model_qint8_avx512_vnni.onnx','config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json'}
        if (manifest.get('repo')!='intfloat/multilingual-e5-small' or not manifest.get('revision')
                or manifest.get('model_file')!='onnx/model_qint8_avx512_vnni.onnx'
                or not isinstance(manifest.get('files'),list) or {r.get('file')for r in manifest['files']}!=required):
            raise SemanticRecallUnavailable('local_embedding_model_not_allowlisted')
        for item in manifest.get('files',[]):
            candidate=(path/item['file']).resolve()
            if not candidate.is_relative_to(path) or not candidate.is_file() or sha256(candidate.read_bytes()).hexdigest()!=item['sha256']:
                raise SemanticRecallUnavailable('local_embedding_model_integrity_failed')
        try:
            from fastembed import TextEmbedding
            from fastembed.common.model_description import ModelSource,PoolingType
            import onnxruntime
        except ImportError:raise SemanticRecallUnavailable('local_fastembed_not_installed') from None
        try:
            onnxruntime.disable_telemetry_events()
            name='trust-judge/local-multilingual-e5-small-'+manifest['revision'][:12]
            registered=[row for row in TextEmbedding.list_supported_models()if row['model']==name]
            if registered:
                if registered[0]['dim']!=384 or registered[0]['model_file']!=manifest['model_file']:
                    raise ValueError('model_registry_binding_conflict')
            else:
                TextEmbedding.add_custom_model(model=name,pooling=PoolingType.MEAN,
                    normalization=True,sources=ModelSource(hf='intfloat/multilingual-e5-small'),dim=384,model_file=manifest['model_file'])
            self.model=TextEmbedding(model_name=name,specific_model_path=str(path),cache_dir=str(path/'runtime-cache'),
                local_files_only=True,threads=threads,providers=['CPUExecutionProvider'])
        except Exception:raise SemanticRecallUnavailable('local_embedding_model_load_failed') from None
        self.model_fingerprint=sha256(_canonical(manifest)).hexdigest()
        self.manifest=manifest

    def embed_passages(self,texts,*,batch_size=32):
        return list(self.model.embed(['passage: '+text for text in texts],batch_size=batch_size))

    def embed_queries(self,texts,*,batch_size=32):
        return list(self.model.embed(['query: '+text for text in texts],batch_size=batch_size))
