from functools import partial
import os
from typing import Generator, List, Optional
from docarray import DocumentArray, Document
import librosa

STORAGE = os.getenv('DOCARRAY_STORAGE_BACKEND')
STORE_CONF = {'host': os.getenv('DOCARRAY_STORAGE_HOST'),
              'port': os.getenv('DOCARRAY_STORAGE_PORT'),
              'index_name': os.getenv('DOCARRAY_STORAGE_INDEX')}


def _extend_docstore(storage: str, config: dict, docs_kwargs: List[dict]):
    with DocumentArray(storage, config) as da:
        da.extend([Document(**kwargs) for kwargs in docs_kwargs])
        return len(da)

def _add_docs(uris: List[str], labels: List[str], n_dim: int, storage: str = STORAGE, config: dict = STORE_CONF):
    assert len(uris) == len(labels)
    config['n_dim'] = n_dim
    metadata = [{'label': l} for l in labels]
    docs_kwargs = [{'uri': u, 'tags': m} for u, m in zip(uris, metadata)]
    return _extend_docstore(storage, config, docs_kwargs)

def _get_doc_by_uri(uri: str, n_dim: int, storage: str = STORAGE, config: dict = STORE_CONF):
    config['n_dim'] = n_dim
    with DocumentArray(storage, config) as da:
        return da.find({'uri': {'$eq': uri}})

def _get_array_by_uris(uris: List[str], n_dim: int, storage: str = STORAGE, config: dict = STORE_CONF):
    config['n_dim'] = n_dim
    with DocumentArray(storage, config) as da:
        return da.find({'uri': {'$in': uris}})

def _doc_audio_to_tensor(doc: Document, in_sr: int = 8000, out_sr: int = 16000) -> Document:
    y, sr = librosa.load(doc.uri, sr=in_sr)
    if in_sr != out_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=out_sr)
    doc.tensor = y
    return doc

def _array_audio_to_tensor(da: DocumentArray, in_sr: int, out_sr: int) -> DocumentArray:
    return da.apply(partial(_doc_audio_to_tensor, in_sr=in_sr, out_sr=out_sr))

def _iter_audio_batch(da: DocumentArray, in_sr: int, out_sr: int, batch: int = 16) -> Generator[DocumentArray, None, None]:
    _func = partial(_doc_audio_to_tensor, in_sr=in_sr, out_sr=out_sr)
    def wrap_func(arr):
        for doc in arr:
            doc = _func(doc)
        return arr
    return da.apply_batch(wrap_func, batch_size=batch)