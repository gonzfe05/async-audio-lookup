from typing import Callable, List
import torch
from torch.utils.data import Dataset
# import torchdatasets as td
from lookup.doc_utils import _get_array_by_uris, _doc_audio_to_tensor

CHARS_TO_EXCLUDE = ['/','_','.','-']

class SegmentDataset(Dataset):
    def __init__(self, uris: List[str], in_sr: int, out_sr: int):
        # super().__init__()
        self.uris = uris
        self.doc_array = _get_array_by_uris(uris)
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.word_delimiter_token = "|"
        self.vocab = self.build_vocab()
        self.in_sr = in_sr
        self.out_sr = out_sr

    def __getitem__(self, idx):
        doc = self.doc_array[idx]
        doc = _doc_audio_to_tensor(doc, in_sr=self.in_sr, out_sr=self.out_sr)
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
    def __init__(self, uris: List[str], in_sr: int, out_sr: int, preprocess_func: Callable):
        self.segment_data = SegmentDataset(uris, in_sr, out_sr)
        self.vocab = self.segment_data.vocab
        self.preprocess_func = preprocess_func

    def __getitem__(self, idx):
        return self.preprocess_func(self.segment_data[idx])
    
    def __len__(self):
        return len(self.segment_data)