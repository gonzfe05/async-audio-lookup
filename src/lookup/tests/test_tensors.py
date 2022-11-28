import pytest

from lookup.doc_utils import DocArrayHandler

from pydub.generators import WhiteNoise


@pytest.fixture(scope='function')
def white_noise(tmp_path):
    uris = []
    for i in range(3):
        file_path = tmp_path / f"{i}.wav"
        uris.append(str(file_path))
        audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1000)
        audio_segment.export(file_path, format='wav')
    yield uris


def test_array_audio_to_tensor(white_noise, start_storage):
    da_h = DocArrayHandler('redis', index_name='audio_to_tensor')
    n_added = da_h.add_docs(white_noise, labels=['a', 'b', 'c'])
    da = da_h.get_docs_by_uris(white_noise)
    da = da_h.array_audio_to_tensor(da, 8000, 16000)
    print(da.tensors.shape)
    assert len(da.tensors.shape) == 2
    assert da.tensors.shape[0] == n_added
    assert da.tensors.shape[1] == 16000



def test_get_batch_docs(white_noise, start_storage):
    print(start_storage)
    da_h = DocArrayHandler(storage = 'redis', index_name='get_batch') 
    n_added = da_h.add_docs(white_noise, labels=['a', 'b', 'c'])
    da = da_h.get_docs_by_uris(white_noise)
    batched_da = da_h.batch_audio_to_tensor(da, in_sr=8000, out_sr=16000, batch=3)
    print(da.tensors.shape)
    assert len(da.tensors.shape) == 2
    assert da.tensors.shape[0] == n_added
    assert da.tensors.shape[1] == 16000