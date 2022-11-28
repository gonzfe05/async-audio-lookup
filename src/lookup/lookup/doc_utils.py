from functools import partial
import os
from typing import Generator, List, Optional
from docarray import DocumentArray, Document
import librosa
import logging
# from celery.utils.log import get_task_logger
# logger = logging.getLogger(__name__)

# STORAGE = os.getenv('DOCARRAY_STORAGE_BACKEND')
# STORE_CONF = {'host': os.getenv('DOCARRAY_STORAGE_HOST'), 'port': os.getenv('DOCARRAY_STORAGE_PORT'), 'n_dim': int(os.getenv('DOCARRAY_NDIM', 0)), 'index_name': os.getenv('DOCARRAY_STORAGE_INDEX'), 'columns': {'uri': 'str', 'label': 'str'}}

class DocArrayHandler(object):
    def __init__(self, storage: str = 'redis', port: int = 6379, n_dim: int = 1568, index_name: str = '1') -> None:
        self.storage = storage
        self.store_conf = {'n_dim': n_dim, 'port': port, 'index_name': index_name, 'columns': {'uri': 'str', 'label': 'str'}}

    def _extend_docstore(self, docs_kwargs: List[dict]):
        with DocumentArray(storage=self.storage, config=self.store_conf) as da:
            prev_n = len(da)
            da.extend([Document(**kwargs) for kwargs in docs_kwargs])
            return len(da) - prev_n

    def add_docs(self, uris: List[str], labels: List[str]):
        assert isinstance(uris, list), f"uris must be a list of strings: {uris}"
        assert len(uris) == len(labels)
        docs_kwargs = [{'tags': {'label': l, 'uri': self._redis_tag_escaper(u)}, 'uri': u} for l, u in zip(labels, uris)]
        return self._extend_docstore(docs_kwargs)

    def del_docs(self, uris: List[str]):
        assert isinstance(uris, list), f"uris must be a list of strings: {uris}"
        with DocumentArray(storage=self.storage, config=self.store_conf) as da:
            start = len(da)
            uris = [self._redis_tag_escaper(u) for u in uris]
            mask = [d.tags['uri'] in uris for d in da]
            del da[mask]
            end = len(da)
        return {'removed': start - end, 'queried': start}

    def embed_docs(self, ids: List[str], embeddings) -> None:
        # logger = get_task_logger(__name__)
        da = DocumentArray(storage=self.storage, config=self.store_conf)
        # mask = [d.uri in uris for d in da]
        # da = self.get_docs_by_uris(uris)
        # ids = [d.id for d in da]
        with da:
            da[ids, 'embedding'] = embeddings
            
    def _get_da(self) -> DocumentArray:
        return DocumentArray(storage=self.storage, config=self.store_conf)

    def get_docs_by_uris(self, uris: List[str]) -> DocumentArray:
        assert isinstance(uris, list), f"uris must be a list of strings: {uris}"
        with DocumentArray(storage=self.storage, config=self.store_conf) as da:
            uris = [self._redis_tag_escaper(u) for u in uris]
            return da.find(filter=f"@uri: {'|'.join(uris)}")

    def _doc_audio_to_tensor(self, doc: Document, in_sr: int = 8000, out_sr: int = 16000) -> Document:
        y, sr = librosa.load(doc.uri, sr=in_sr)
        if in_sr != out_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=out_sr)
        doc.tensor = y
        return doc

    def array_audio_to_tensor(self, da: DocumentArray, in_sr: int, out_sr: int) -> DocumentArray:
        return da.apply(partial(self._doc_audio_to_tensor, in_sr=in_sr, out_sr=out_sr))

    def batch_audio_to_tensor(self, da: DocumentArray, in_sr: int, out_sr: int, batch: int = 16) -> Generator[DocumentArray, None, None]:
        _func = partial(self._doc_audio_to_tensor, in_sr=in_sr, out_sr=out_sr)
        def wrap_func(arr):
            for doc in arr:
                doc = _func(doc)
            return arr
        return da.apply_batch(wrap_func, batch_size=batch)
    
    def _redis_tag_escaper(self, tag: str) -> str:
        special_chars_map = {',': '\,',
                            '.': '\.',
                            '<': '\<',
                            '>': '\>',
                            '{': '\{',
                            '}': '\}',
                            '[': '\[',
                            ']': '\]',
                            '"': '\"',
                            "'": "\'",
                            ':': '\:',
                            ';': '\;',
                            '!': '\!',
                            '@': '\@',
                            '#': '\#',
                            '$': '\$',
                            '%': '\%',
                            '^': '\^',
                            '&': '\&',
                            '*': '\*',
                            '(': '\(',
                            ')': '\)',
                            '-': '\-',
                            '+': '\+',
                            '=': '\=',
                            '~': '\~',
                            ' ': '\ '
                            }
        return ''.join(special_chars_map.get(c, c) for c in tag)
