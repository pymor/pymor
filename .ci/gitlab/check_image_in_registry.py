#!/usr/bin/env python3

import os
import sys

import gitlab

image, tag = sys.argv[1], sys.argv[2]

print(f'Checking if {image}:{tag} is available in gitlab registry ... ', end='')

gl = gitlab.Gitlab('https://zivgitlab.uni-muenster.de', private_token=os.environ['GITLAB_API_RO'])
gl.auth()

pymor_id = 976
pymor = gl.projects.get(pymor_id)

for repo in pymor.repositories.list(all=True):
    if repo.name != image:
        continue
    try:
        tag = repo.tags.get(id=tag)
        print('yes!')
        sys.exit(0)
    except gitlab.exceptions.GitlabGetError:
        break

print('no!')
sys.exit(1)
