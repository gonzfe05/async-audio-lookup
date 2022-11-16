from functools import partial
import os
from typing import Generator, List, Optional
from docarray import DocumentArray, Document
import librosa

STORAGE = os.getenv('DOCARRAY_STORAGE_BACKEND')
STORE_CONF = {'host': os.getenv('DOCARRAY_STORAGE_HOST'), 'port': os.getenv('DOCARRAY_STORAGE_PORT'), 'n_dim': int(os.getenv('DOCARRAY_NDIM', 0)), 'index_name': os.getenv('DOCARRAY_STORAGE_INDEX'), 'columns': {'uri': 'str', 'label': 'str'}}


def _extend_docstore(docs_kwargs: List[dict]):
    with DocumentArray(storage=STORAGE, config=STORE_CONF) as da:
        da.extend([Document(**kwargs) for kwargs in docs_kwargs])
        return len(da)

def add_docs(uris: List[str], labels: List[str]):
    assert len(uris) == len(labels)
    metadata = [{'label': l, 'uri': u} for l, u in zip(labels, uris)]
    docs_kwargs = [{'tags': t, 'uri': t['uri']} for t in metadata]
    return _extend_docstore(docs_kwargs)

def del_docs(uris: List[str]):
    with DocumentArray(storage=STORAGE, config=STORE_CONF) as da:
        start = len(da)
        mask = [d.tags['uri'] in uris for d in da]
        del da[mask]
        end = len(da)
    return {'removed': start - end, 'queried': start}

def _get_da() -> DocumentArray:
    return DocumentArray(storage=STORAGE, config=STORE_CONF)

def _get_doc_by_uri(uri: str) -> DocumentArray:
    with DocumentArray(storage=STORAGE, config=STORE_CONF) as da:
        return da.find(filter={'uri': {'$eq': uri}})

def _get_array_by_uris(uris: List[str]):
    with DocumentArray(storage=STORAGE, config=STORE_CONF) as da:
        #TODO: use find method
        mask = [d.uri in uris for d in da]
        return da[mask]

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