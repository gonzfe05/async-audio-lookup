from numpy.random import rand
from numpy.testing import assert_array_equal
import pytest

from docarray.math.ndarray import to_numpy_array

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

@pytest.mark.parametrize('uris, labels, n_dim, index',
                        [
                            (['uri1.wav'], ['a'], 3, 'emb_one_dim'),
                            (['uri1.wav', 'uri2.wav'], ['a', 'b'], 3, 'emb_mult_dim'),
                        ])
def test_array_audio_to_tensor(uris, labels, n_dim, index, start_storage):
    da_h = DocArrayHandler('redis', n_dim=n_dim, index_name=index)
    embedding = rand(len(uris), n_dim)
    n_added = da_h.add_docs(uris, labels)
    da = da_h.get_docs_by_uris(uris)
    ids = [d.id for d in da]
    da_h.embed_docs(ids, embedding)
    da = da_h.get_docs_by_uris(uris)
    print(da.embeddings.shape)
    assert da[ids, 'embedding'].tolist() == embedding.tolist(), f"{da.embeddings} != {embedding}"
    if len(uris) > 1:
        embedding_2 = rand(1, n_dim)
        da_h.embed_docs([ids[0]], embedding_2)
        da = da_h.get_docs_by_uris(uris)
        assert da[ids, 'embedding'].tolist()[0] == embedding_2.tolist()[0], f"{da[0].embedding} != {embedding_2}"
        assert da[ids, 'embedding'].tolist()[1:] == embedding.tolist()[1:], f"{da.embeddings[1:]} != {embedding[1:]}"

    