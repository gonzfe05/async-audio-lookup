from typing import Callable, List, Callable
import os
from docarray import DocumentArray, Document
from typing import List
import torch
from torch.utils.data import Dataset, IterableDataset
# import torchdatasets as td
from lookup.doc_utils import DocArrayHandler #_get_array_by_uris, _doc_audio_to_tensor

CHARS_TO_EXCLUDE = ['/','_','.','-']

class SegmentDataset(Dataset):
    def __init__(self, uris: List[str], in_sr: int, out_sr: int, docarray_handler: DocArrayHandler):
        # super().__init__()
        self.uris = uris
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.word_delimiter_token = "|"
        self.in_sr = in_sr
        self.out_sr = out_sr
        self.da_h = docarray_handler
        self.doc_array = self.da_h.get_docs_by_uris(uris)
        self.uri_ix_map = {d.uri: ix for ix, d in enumerate(self.doc_array)}
        self.vocab = self.build_vocab()

    def __getitem__(self, idx):
        doc = self.doc_array[self.uri_ix_map[self.uris[idx]]]
        doc = self.da_h._doc_audio_to_tensor(doc, in_sr=self.in_sr, out_sr=self.out_sr)
        audio = {'array': torch.tensor(doc.tensor), 'sampling_rate': self.out_sr, 'path': doc.uri}
        return {'file': doc.uri, 'audio': audio, 'text': doc.tags['label']}

    def __len__(self):
        return len(self.doc_array)
    
    def build_vocab(self):
        vocab = []
        for doc in self.doc_array:
            vocab.extend(list(doc.tags['label']))
            vocab = list(set(vocab))
        vocab = {v: ix for ix, v in enumerate(vocab) if v not in CHARS_TO_EXCLUDE}
        # To make it clearer that " " has its own token class, we give it a more visible character |
        vocab[self.word_delimiter_token] = vocab.get(" ", len(vocab))
        if vocab.get(" "):
            del vocab[" "]
        # Add [PAD] token for CTC annd [UNK] for oov tokens
        vocab[self.unk_token] = len(vocab)
        vocab[self.pad_token] = len(vocab)
        return vocab

class CustomDataset(Dataset):
    def __init__(self, uris: List[str], in_sr: int, out_sr: int, preprocess_func: Callable, da: DocArrayHandler):
        self.segment_data = SegmentDataset(uris, in_sr, out_sr, da)
        self.vocab = self.segment_data.vocab
        self.preprocess_func = preprocess_func

    def __getitem__(self, idx):
        return self.preprocess_func(self.segment_data[idx])
    
    def __len__(self):
        return len(self.segment_data)


class AudioDataset(IterableDataset):
    def __init__(self, uris: List[str], out_sr: int, docarray_handler_kwargs: dict = None, prep_func: Callable = None) -> None:
        self.uris = uris
        self.out_sr = out_sr
        self.da_h = DocArrayHandler(**docarray_handler_kwargs) if docarray_handler_kwargs else DocArrayHandler()
        da = self.da_h.get_docs_by_uris(uris)
        self.data_path = self._store(da, path='.', name='da')
        self.prep_func = prep_func if prep_func else lambda x: x
        self.dataloader = self._get_dataloader(self.data_path, self.prep_func)

    def _store(self, da: DocumentArray, path: str, name: str):
        data_path = os.path.join(path, f'{name}.protobuf.gz')
        da.save_binary(data_path)
        return data_path
    
    def _format_docarray(self, da: DocumentArray):
        da = self.da_h.array_audio_to_tensor(da, in_sr=8000, out_sr=self.out_sr)
        docs = []
        for doc in da:
            audio = {'array': torch.tensor(doc.tensor), 'sampling_rate': self.out_sr, 'path': doc.uri}
            docs.append({'file': doc.uri, 'audio': audio, 'text': doc.tags['label']})
        return docs
    
    def _get_dataloader(self, data_path: str, prep_func: Callable):
        func = lambda x: prep_func(self._format_docarray(x))
        return DocumentArray.dataloader(data_path, func=func, batch_size=64, num_worker=4)

    def __iter__(self):
        return self.dataloader