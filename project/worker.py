import os
import time
from typing import List, Optional
from pydantic import BaseModel
from lookup.doc_utils import add_docs as _add_docs, del_docs as _del_docs, _get_doc_by_uri, _doc_audio_to_tensor, _get_array_by_uris, _array_audio_to_tensor
from lookup.wav2vec import get_data_collator, load_w2v, load_trainer, w2v_data_loader, get_logits as _get_logits

from uvicorn import Config

from celery import Celery

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


class Doc(BaseModel):
    uri: str
    tags: Optional[dict]


@celery.task(name="create_task")
def create_task(task_type: int):
    time.sleep(task_type)
    return True


@celery.task(name="add_docs")
def add_docs(docs: List[Doc]):
    docs = [Doc.parse_obj(d) for d in docs]
    uris = [d.uri for d in docs]
    labels = [d.tags['label'] for d in docs]
    return _add_docs(uris, labels)


@celery.task(name="del_docs")
def del_docs(docs: List[Doc]):
    docs = [Doc.parse_obj(d) for d in docs]
    uris = [d.uri for d in docs]
    return _del_docs(uris)

@celery.task(name="get_doc")
def get_doc(uri: str):
    return _get_doc_by_uri(uri)

@celery.task(name="get_doc_array")
def get_doc_array(uris: List[str]):
    return _get_array_by_uris(uris)

@celery.task(name="get_doc_audio")
def get_doc_audio(uri: str, in_sr: int = 8000, out_sr: int = 16000):
    doc = get_doc(uri)[0]
    return _doc_audio_to_tensor(doc, in_sr, out_sr)

@celery.task(name="get_array_audio")
def array_audio_to_tensor(uris: List[str], in_sr: int = 8000, out_sr: int = 16000):
    da = _get_array_by_uris(uris)
    return _array_audio_to_tensor(da, in_sr, out_sr)
    

@celery.task(name="train_encoder")
def train_encoder(docs: List[dict], checkpoint_path: str, in_sr: int, out_sr: int):
    docs = [Doc.parse_obj(d) for d in docs]
    docs = [d for d in docs if d.tags.get('label')]
    uris = [d.uri for d in docs]
    dataset, processor = w2v_data_loader(uris, in_sr, out_sr)
    model = load_w2v(processor)
    data_collator = get_data_collator(processor)
    trainer = load_trainer(model, data_collator, dataset, dataset, processor, epochs=1, output_dir = checkpoint_path)
    trainer.train()
    return trainer.state.log_history

@celery.task(name="get_logits")
def get_logits(batch, model, device: str = "cpu"):
    return get_logits(batch['input_values'], model, device)


@celery.task(name="get_doc_embeddings")
def get_docs_embeddings(docs: List[dict], in_sr: int = 8000, out_sr: int = 16000):
    raise NotImplementedError
    docs = [Doc.parse_obj(d) for d in docs]
    uris = [d.uri for d in docs]
    dataset, processor = w2v_data_loader(uris, in_sr, in_sr)
    model = load_w2v(processor)
    get_logits(dataset, )
