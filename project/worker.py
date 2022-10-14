import os
import time
from typing import Generator, List, Optional

from uvicorn import Config

from celery import Celery

celery = Celery(__name__)
celery.conf.broker_url = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
celery.conf.result_backend = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")


@celery.task(name="create_task")
def train_embedding(uri_list: List[str]):
    time.sleep(int(uri_list) * 10)
    return True
