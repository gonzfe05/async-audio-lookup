import pytest
from lookup.dataset import CustomDataset, SegmentDataset
from lookup.doc_utils import Document
from pydub.generators import WhiteNoise
import numpy as np
from torch import is_tensor


def test_segment_dataset(tmp_path, mocker):
    uris = []
    for i in range(3):
        file_path = tmp_path / f"{i}.wav"
        uris.append(str(file_path))
        audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1)
        audio_segment.export(file_path, format='wav')
    mock_doc_array = lambda uris, n_dim: [Document(uri=i, tags={'label': str(i)}) for i in uris]
    mock = mocker.patch('lookup.dataset._get_array_by_uris', mock_doc_array)
    segment_dataset = SegmentDataset(uris, 512, 8000, 16000)
    assert len(segment_dataset) == 3
    for seg in segment_dataset:
        assert isinstance(seg['file'], str)
        assert is_tensor(seg['audio']['array'])
        assert len(seg['audio']['array'].shape) == 1
        assert seg['audio']['sampling_rate'] == 16000
        assert isinstance(seg['text'], str)
    assert isinstance(segment_dataset.vocab, dict)
    for k, v in segment_dataset.vocab.items():
        assert isinstance(k, str)
        assert isinstance(v, int)


def test_custom_dataset(tmp_path, mocker):
    uris = []
    for i in range(3):
        file_path = tmp_path / f"{i}.wav"
        uris.append(str(file_path))
        audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1)
        audio_segment.export(file_path, format='wav')
    mock_doc_array = lambda uris, n_dim: [Document(uri=i, tags={'label': str(i)}) for i in uris]
    mock = mocker.patch('lookup.dataset._get_array_by_uris', mock_doc_array)
    def prep_func(_input):
        _input['audio']['array2'] = _input['audio']['array'] * 0
        return _input
    segment_dataset = CustomDataset(uris, 512, 8000, 16000, prep_func)
    assert len(segment_dataset) == 3
    for seg in segment_dataset:
        assert isinstance(seg['file'], str)
        assert is_tensor(seg['audio']['array'])
        assert len(seg['audio']['array'].shape) == 1
        assert is_tensor(seg['audio']['array2'])
        assert len(seg['audio']['array2'].shape) == 1
        assert seg['audio']['sampling_rate'] == 16000
        assert isinstance(seg['text'], str)
    assert isinstance(segment_dataset.vocab, dict)
    for k, v in segment_dataset.vocab.items():
        assert isinstance(k, str)
        assert isinstance(v, int)