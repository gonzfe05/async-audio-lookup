import tempfile
import os
import pytest
import random

cur_dir = os.path.dirname(os.path.abspath(__file__))
compose_yml = os.path.abspath(
    os.path.join(cur_dir, 'assets', 'docker-compose.yml')
)


@pytest.fixture(autouse=True)
def tmpfile(tmpdir):
    _tmpfile = f'docarray_test_{next(tempfile._get_candidate_names())}.db'
    return tmpdir / _tmpfile


@pytest.fixture(scope='module')
def start_storage():
    name = str(random.randint(0, 10000))
    os.system(
        f"docker-compose -f {compose_yml} -p {name} --project-directory . up  --build -d "
        # f"--remove-orphans --no-recreate"
    )

    
    yield os.system("docker ps")
    os.system(
        f"docker-compose -f {compose_yml} -p {name} --project-directory . down "
        # f"--remove-orphans"
    )


@pytest.fixture(scope='session')
def set_env_vars(request):
    _old_environ = dict(os.environ)
    os.environ.update(request.param)
    yield
    os.environ.clear()
    os.environ.update(_old_environ)