import pytest
from lookup.doc_utils import DocArrayHandler
from lookup.dataset import CustomDataset, SegmentDataset, AudioDataset
from pydub.generators import WhiteNoise
from random import choice

@pytest.fixture(scope='function')
def white_noise(tmp_path):
    uris = []
    for i in range(3):
        file_path = tmp_path / f"{i}.wav"
        uris.append(str(file_path))
        audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1000)
        audio_segment.export(file_path, format='wav')
    yield uris


def test_segment_dataset(white_noise, start_storage):
    da_h = DocArrayHandler(storage = 'redis', index_name='segment_dataset') 
    labels = [choice(['one label', 'another label']) for _ in white_noise]
    n_added = da_h.add_docs(white_noise, labels)
    dataset = SegmentDataset(white_noise, in_sr=8000, out_sr=16000, docarray_handler=da_h)
    assert len(dataset) == len(white_noise), f"dataset: {dataset}, uris: {white_noise}"
    for i in range(len(dataset)):
        example = dataset[i]
        assert example['file'] == white_noise[i]
        assert example['audio']['sampling_rate'] == 16000
        assert example['audio']['array'].shape[0] == 16000
        assert example['text'] == labels[i]


def test_custom_dataset(white_noise, start_storage):
    da_h = DocArrayHandler(storage = 'redis', index_name='segment_dataset') 
    labels = [choice(['one label', 'another label']) for _ in white_noise]
    n_added = da_h.add_docs(white_noise, labels)
    prep_func = lambda data: data['audio']['array']
    dataset = CustomDataset(white_noise, in_sr=8000, out_sr=16000, preprocess_func=prep_func, da=da_h)
    assert len(dataset) == len(white_noise), f"dataset: {dataset}, uris: {white_noise}"
    for i in range(len(dataset)):
        example = dataset[i]
        assert example.shape[0] == 16000


def test_audio_dataset(white_noise, start_storage):
    da_h = DocArrayHandler(storage = 'redis', index_name='iter_dataset') 
    labels = [choice(['one label', 'another label']) for _ in white_noise]
    n_added = da_h.add_docs(white_noise, labels)
    assert n_added == len(white_noise)
    kwargs = {'index_name': 'iter_dataset'}

    dataset = AudioDataset(uris=white_noise, out_sr=16000, docarray_handler_kwargs=kwargs)
    uris = white_noise.copy()
    for ix, examples in enumerate(dataset):
        assert len(examples) == len(white_noise)
        for jx, example in enumerate(examples):
            assert example['file'] in uris
            i = uris.index(example['file'])
            assert example['text'] == labels[i]
            assert example['audio']['sampling_rate'] == 16000
            assert example['audio']['array'].shape[0] == 16000
    assert ix == 0
    
    def prep_func(examples):
        res = []
        for example in examples:
            res.append({'text': f"{example['text']}_post"})
        return res
    
    dataset = AudioDataset(uris=white_noise, out_sr=16000, docarray_handler_kwargs=kwargs, prep_func=prep_func)
    for ix, examples in enumerate(dataset):
        assert len(examples) == len(white_noise)
        for jx, example in enumerate(examples):
            assert example['text'] == f"{labels[jx]}_post"