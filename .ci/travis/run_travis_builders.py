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
    env = { 'PYMOR_PYTEST_MARKER': marker,
        'TRAVIS_REPO_SLUG':'pymor/pymor',
        'TRAVIS_PULL_REQUEST': 'false',
        'TRAVIS_COMMIT': commit,
    }

    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        subprocess.check_call(['git', 'clone', clone_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir('pymor')
        volumes = { os.getcwd(): { 'bind': '/src',  'mode': 'rw'}}
        subprocess.check_call(['git', 'checkout', commit], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            container = client.containers.run(image, environment=env,
                                            stderr=True, detach=True,
                                            volumes=volumes,
                                            command='./.ci/travis/script.bash')
        except docker.errors.ContainerError as err:
            return False, err
        exit = container.wait()
        if exit:
            return False, container.logs(stream=False, stderr=True, stdout=True)
        return True, ''


docker_tags = env.list('PYMOR_DOCKER_TAG', default=['3.6', '3.7'])
pytest_marker = env.list('PYMOR_PYTEST_MARKER', default=["None", 'PIP_ONLY', 'MPI'])
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
variations = list(product(docker_tags, pytest_marker))
if '3.6' in docker_tags and len(pytest_marker) > 1:
    variations += [('3.6', 'NUMPY')]
with TemporaryDirectory() as clone_tmp:
    clone_dir = os.path.join(clone_tmp, 'pymor')
    subprocess.check_call(['git', 'clone', os.getcwd(), clone_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_configs = partial(_run_config, clone_dir=clone_dir, commit=commit)
    results = dict()
    for tm in variations:
        results[tm] = run_configs(tm)
for tm, (success, msg) in results.items():
    print('Variation {0[0]} - {0[1]}: {1}'.format(tm, 'success' if success else 'failed'))
for tm, (success, msg) in results.items():
    if not success:
        print('\nFaillogs {0[0]} - {0[1]}: \n{1}'.format(tm, msg))
