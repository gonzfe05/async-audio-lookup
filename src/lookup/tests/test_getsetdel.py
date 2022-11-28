import pytest

from lookup.doc_utils import DocArrayHandler


@pytest.mark.parametrize('uris, labels',
                        [
                            (['uri1.wav'], ['a']),
                            (['uri2.wav'], ['b']),
                            (['uri1.wav', 'uri2.wav'], ['a', 'b']),
                            (['uri1.wav', 'uri2.wav'], ['a']),
                        ])
def test_add_docs(uris, labels, start_storage):
    print(start_storage)
    da_h = DocArrayHandler(storage = 'redis', index_name='add') 
    if len(uris) == len(labels) and len(set(uris)) == len(uris):
        n_added = da_h.add_docs(uris, labels)
        assert n_added == len(uris)
    if len(uris) == len(labels) and len(set(uris)) != len(uris):
        n_added = da_h.add_docs(uris, labels)
        assert n_added == len(set(uris))
    if len(uris) != len(labels):
        with pytest.raises(Exception) as e_info:
            da_h.add_docs(uris, labels)


@pytest.mark.parametrize('uris, labels, index',
                        [
                            (['uri1.wav', 'uri2.wav'], ['a', 'b'], 'get_1'),
                            (['temp/uri1.wav', 'temp/uri2.wav'], ['a', 'b'], 'get_2'),
                            (['/large/path/with-escaped-chars/uri 1.wav', '/large/path/with-escaped-chars/uri 2.wav'], ['a', 'b'], 'get_3'),
                        ])
def test_get_docs(uris, labels, index, start_storage):
    print(start_storage)
    da_h = DocArrayHandler(storage = 'redis', index_name=index) 
    n_added = da_h.add_docs(uris, labels)
    assert n_added == len(uris)
    retrieved = da_h.get_docs_by_uris(uris)
    assert len(retrieved) == n_added
    assert set(uris) == {da.uri for da in retrieved}
    if len(uris) > 0:
        retrieved = da_h.get_docs_by_uris([uris[0]])
        assert len(retrieved) == 1
        assert uris[0] == retrieved[0, 'uri']


@pytest.mark.parametrize('uris,labels,index',
                        [
                            (['uri1.wav'], ['a'], 'del_1'),
                            (['uri1.wav', 'uri2.wav'], ['a', 'b'], 'del_2')
                        ])
def test_del_docs(uris, labels, index, start_storage):
    print(start_storage)
    da_h = DocArrayHandler(storage = 'redis', index_name=index) 
    n_added = da_h.add_docs(uris, labels)
    assert n_added == len(uris)
    deleted = da_h.del_docs(uris)
    assert deleted['queried'] == len(uris)
    assert deleted['removed'] == len(uris)

    if len(uris) > 1:
        n_added = da_h.add_docs(uris, labels)
        assert n_added == len(uris)
        deleted = da_h.del_docs([uris[0]])
        assert deleted['queried'] == len(uris)
        assert deleted['removed'] == 1