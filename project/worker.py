import os
from pickle import LIST
import time
from typing import List, Optional

from uvicorn import Config
from docarray import DocumentArray, Document

from celery import Celery

STORAGE = os.getenv('DOCARRAY_STORAGE_BACKEND')
assert STORAGE
CONFIG = {'host': os.getenv('DOCARRAY_STORAGE_HOST'),
          'port': os.getenv('DOCARRAY_STORAGE_PORT'),
          'index_name': os.getenv('DOCARRAY_STORAGE_INDEX')}

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")

def _add_doc(uris: List[str], labels: List[str], n_dim: int, storage: Optional[str] = STORAGE, config: dict = CONFIG):
    assert len(uris) == len(labels)
    config['n_dim'] = n_dim
    with DocumentArray(storage, config) as da:
        da.extend([Document(uri=u, tags={'label': l}) for u, l in zip(uris, labels)])
        return len(da)

def _get_doc(uri: str, n_dim: int, storage: Optional[str] = STORAGE, config: dict = CONFIG):
    config['n_dim'] = n_dim
    da = DocumentArray(storage, config)
    with da as d:

@celery.task(name="create_task")
def train_embedding(uri_list: List[str]):
    time.sleep(int(uri_list) * 10)
    return True
