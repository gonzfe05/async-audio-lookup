import os
import time
from typing import Generator, List, Optional
from pydantic import BaseModel
from lookup.doc_utils import add_docs as _add_docs

from uvicorn import Config

from celery import Celery

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")

class doc(BaseModel):
    uri: str
    tags: dict

@celery.task(name="create_task")
def create_task(task_type: int):
    time.sleep(task_type)
    return True

@celery.task(name="add_docs")
def add_docs(docs: List[doc]):
    docs = [doc.parse_obj(d) for d in docs]
    uris = [d.uri for d in docs]
    labels = [d.tags['label'] for d in docs]
    return _add_docs(uris, labels, 512)