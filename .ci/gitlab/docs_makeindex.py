#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from jinja2 import Template


tpl = '''
<!doctype html>
<head>
<title>pyMOR Documentation</title>
</head>
<body>
<h1>Available versions</h1>

<ul>
    {%- for subdir in subdirs -%}
    <li>
      <a href="{{ subdir }}/index.html">{{ subdir }}</a>
    </li>
    {%- endfor %}
</ul>

</body>
</html>
'''


def _make_list(root, name='master'):
    subdirs = []
    for it in root.iterdir():
        if it.is_dir() and (it / 'index.html').is_file():
            subdirs.append(it.name)
    return sorted(subdirs)


def make_index(path):
    path = path.resolve()
    with (path / 'list.html').open('wt') as out:
        out.write(Template(tpl).render(subdirs=_make_list(path)))


if __name__ == '__main__':
    try:
        path = Path(sys.argv[1])
    except IndexError:
        path = Path().cwd()
    make_index(path)
