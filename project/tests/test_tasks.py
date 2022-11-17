from pathlib import Path   
import json
from typing import List
import numpy as np

from unittest.mock import patch, call
from pydub.generators import WhiteNoise
from docarray import DocumentArray, Document
from lookup.doc_utils import _get_da, _doc_audio_to_tensor
from worker import create_task, add_docs, get_doc, del_docs, train_encoder , get_doc_audio, get_doc_array, array_audio_to_tensor, get_model_embeddings


def test_home(test_app):
    response = test_app.get("/")
    assert response.status_code == 200


def test_create_task():
    assert create_task.run(1)
    assert create_task.run(2)
    assert create_task.run(3)

def create_audio(uri: str):
    audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1000)
    audio_segment.export(uri, format='wav')

def test_add_docs(tmp_path):
    uris = [f'{str(tmp_path)}/{n}.wav' for n in ['aaa', 'bbb', 'ccc']]
    for uri in uris:
        create_audio(uri)
    assert add_docs.run([{'uri': uris[0], 'tags': {'label': 'a'}}]) == 1
    assert add_docs.run([{'uri': uris[1], 'tags': {'label': 'b'}}, {'uri': uris[2], 'tags': {'label': 'c'}}]) == 3

def test_get_doc():
    doc = get_doc.run('aaa.wav')
    assert isinstance(doc, DocumentArray)
    assert len(doc) == 1, doc
    assert Path(doc[0].tags['uri']).name == 'aaa.wav'
    doc = get_doc.run('bbb.wav')
    assert len(doc) == 1, doc
    assert doc[0].tags['label'] == 'b'

def test_get_doc_array():
    doc1 = get_doc.run('aaa.wav')[0]
    doc2 = get_doc.run('bbb.wav')[0]
    doc3 = get_doc.run('ccc.wav')[0]
    uris = [u.uri for u in [doc1, doc2, doc3]]
    da = get_doc_array(uris)
    print(da.summary())
    assert [d.uri for d in da] == uris

def test_get_doc_audio():
    doc = get_doc.run('aaa.wav')[0]
    doc = get_doc_audio.run(doc.uri, 8000, 16000)
    print(doc.tensor.shape)
    assert doc.tensor.shape[0] > 0


def test_array_audio_to_tensor():
    doc1 = get_doc.run('aaa.wav')[0]
    doc2 = get_doc.run('bbb.wav')[0]
    doc3 = get_doc.run('ccc.wav')[0]
    uris = [u.uri for u in [doc1, doc2, doc3]]
    da = array_audio_to_tensor(uris, 8000, 16000)
    print(da.summary())
    assert [d.tensor for d in da]
    assert all(d.tensor.shape[0] > 0 for d in da)

def test_get_model_embeddings():
    doc1 = get_doc.run('aaa.wav')[0]
    doc2 = get_doc.run('bbb.wav')[0]
    doc3 = get_doc.run('ccc.wav')[0]
    uris = [u.uri for u in [doc1, doc2, doc3]]
    tensors = get_model_embeddings(uris, in_sr=8000, out_sr=16000)
    tensors = np.array(tensors)
    assert tensors.shape[-1] > 0

def test_del_docs():
    doc = get_doc.run('aaa.wav')[0]
    assert del_docs.run([{'uri': doc.uri}]) == {'removed': 1, 'queried': 3}
    doc1 = get_doc.run('bbb.wav')[0]
    doc2 = get_doc.run('ccc.wav')[0]
    assert del_docs.run([{'uri': doc1.uri}, {'uri': doc2.uri}]) == {'removed': 2, 'queried': 2}

NUM2WORDS_MAP = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \
             6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \
            11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \
            15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \
            19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \
            50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \
            90: 'Ninety', 0: 'Zero'}

def num2word(n):
    try:
        return NUM2WORDS_MAP[n].lower()
    except KeyError:
        try:
            return NUM2WORDS_MAP[n-n%10].lower() + NUM2WORDS_MAP[n%10].lower()
        except KeyError:
            return 'Number out of range'

def create_mock_docs(path: str, n: int) -> List[dict]:
    docs = []
    for i in range(n):
        uri = f"{path}/{i}.wav"
        audio_segment = WhiteNoise(sample_rate=8000).to_audio_segment(duration=1000)
        audio_segment.export(uri, format='wav')
        docs.append({'uri': uri, 'tags': {'label': num2word(i), 'uri': uri}})
    return docs

def test_train_encoder(tmp_path):
    docs = create_mock_docs(str(tmp_path), 32)
    print(docs)
    assert add_docs.run(docs) == len(docs)
    log_history = train_encoder.run(docs, f"{str(tmp_path)}/checkpoint", 8000, 16000)
    print(f"log_history: {log_history}")
    assert len(log_history) == 1
    assert log_history[0]['epoch'] == 1
    assert del_docs.run(docs) == {'queried': 32, 'removed': 32}


@patch("worker.create_task.run")
def test_mock_create_task(mock_run):
    assert create_task.run(1)
    create_task.run.assert_called_once_with(1)

    assert create_task.run(2)
    assert create_task.run.call_count == 2

    assert create_task.run(3)
    assert create_task.run.call_count == 3


def test_create_task_status(test_app):
    response = test_app.post(
        "/tasks/dummy",
        data=json.dumps({"type": 1})
    )
    content = response.json()
    task_id = content["task_id"]
    assert task_id

    response = test_app.get(f"tasks/{task_id}")
    content = response.json()
    assert content == {"task_id": task_id, "task_status": "PENDING", "task_result": None}
    assert response.status_code == 200

    while content["task_status"] == "PENDING":
        response = test_app.get(f"tasks/{task_id}")
        content = response.json()
    assert content == {"task_id": task_id, "task_status": "SUCCESS", "task_result": True}
