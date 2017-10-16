#!/usr/bin/env python3
import logging
import subprocess
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os
from multiprocessing import Pool, cpu_count
from itertools import product
from functools import partial
import docker
from envparse import env

def _run_config(tm, clone_dir, commit):
    tag, marker = tm
    image = 'pymor/testing:{}'.format(tag)
    client = docker.from_env(version='auto')
    client.images.pull(image)
    env = { 'PYTEST_MARKER': marker,
        'TRAVIS_REPO_SLUG':'pymor/pymor',
        'TRAVIS_PULL_REQUEST': 'false',
        'TRAVIS_COMMIT': commit,
    }

    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        subprocess.check_call(['git', 'clone', clone_dir])
        os.chdir('pymor')
        volumes = { os.getcwd(): { 'bind': '/src',  'mode': 'rw'}}
        subprocess.check_call(['git', 'checkout', commit])
        try:
            cont = client.containers.run(image, environment=env,
                                            stderr=True, detach=True,
                                            volumes=volumes,
                                            command='./.ci/travis/script.bash')
        except docker.errors.ContainerError as err:
            logging.error('Failed config: {} - {}'.format(tag, marker))
            logging.error(err)
            return
        for lg in cont.logs(stream=True, stderr=True, stdout=True):
            print(lg.decode())


docker_tags = env.list('PYMOR_DOCKER_TAG', default=['2.7', '3.4', '3.5', '3.6'])
pytest_marker = env.list('PYMOR_PYTEST_MARKER', default=["None", 'PIP_ONLY', 'MPI'])
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
variations = list(product(docker_tags, pytest_marker)) + [('3.6', 'NUMPY')]
with TemporaryDirectory() as clone_tmp:
    clone_dir = os.path.join(clone_tmp, 'pymor')
    subprocess.check_call(['git', 'clone', os.getcwd(), clone_dir])
    run_configs = partial(_run_config, clone_dir=clone_dir, commit=commit)
    for tm in variations:
        run_configs(tm)
