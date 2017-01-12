#!/usr/bin/env python3
import logging
import subprocess
from tempfile import TemporaryDirectory, NamedTemporaryFile
import os
from multiprocessing import Pool, cpu_count
from itertools import product
from functools import partial

env_tpl = '''
PYTEST_MARKER="{}"
TRAVIS_REPO_SLUG={}
TRAVIS_PULL_REQUEST="false"
TRAVIS_COMMIT={}
'''


def _run_config(tm, clone_dir, commit):
    tag, marker = tm
    with TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        subprocess.check_call(['git', 'clone', clone_dir])
        os.chdir('pymor')
        subprocess.check_call(['git', 'checkout', commit])
        image = 'pymor/testing:{}'.format(tag)
        subprocess.check_call(['docker', 'pull', image])
        with NamedTemporaryFile('wt') as envfile:
            envfile.write(env_tpl.format(marker, slug, commit))
            cmd = ['docker', 'run', '--env-file', envfile.name, '-v', '{}:/src'.format(os.getcwd()),
                   image, './.travis.script.bash']
            try:
                _ = subprocess.check_call(cmd)
            except subprocess.CalledProcessError as err:
                logging.error('Failed config: {} - {}'.format(tag, marker))
                logging.error(err)


docker_tags = ['2.7', '3.4', '3.5']
pytest_marker = [None, "PIP_ONLY", "MPI"]
slug = 'pymor/pymor'
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

with TemporaryDirectory() as clone_tmp:
    clone_dir = os.path.join(clone_tmp, 'pymor')
    subprocess.check_call(['git', 'clone', 'https://github.com/{}.git'.format(slug), clone_dir])
    run_configs = partial(_run_config, clone_dir=clone_dir, commit=commit)
    cpus = int(cpu_count()/2)
    with Pool(processes=cpus) as pool:
        _ = pool.map(run_configs, product(docker_tags, pytest_marker))
