# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from html import escape
from importlib import import_module
from inspect import getdoc
from pathlib import Path
from pkgutil import iter_modules
from shutil import rmtree

from cyclopts.docs import generate_rst_docs

import pymordemos

OUT_DIR = Path(__file__).parent / 'source' / 'demos'
rmtree(OUT_DIR, ignore_errors=True)
OUT_DIR.mkdir()

demos = [m.name for m in iter_modules(pymordemos.__path__)]

TAG_VOCABULARY = (
    'stationary', 'instationary', 'linear', 'nonlinear', 'parametric',
    'reduced-basis', 'pod', 'empirical-interpolation', 'system-theoretic', 'data-driven', 'machine-learning',
    'system-identification', 'error-estimation', 'optimization', 'structure-preserving',
    'builtin-discretizer', 'fenics', 'ngsolve', 'scikit-fem',
)

TAG_STYLE = ('<style>'
             '.demo-tag{display: inline-block; margin: 0 .3em .3em 0; padding: .1em .6em; border-radius: 1em;'
             'border: 1px solid #3f51b5; background: #e8eaf6; color: #3f51b5; font-size: .8em;'
             'cursor: pointer; text-decoration: none}'
             '.demo-tag.active{background: #3f51b5; color: #fff}'
             '.demo-tag[data-clear]{margin-left: 1em; border-color: #9e9e9e; background: none; color: #616161}'
             '.demo-tag[data-clear]::before{content: "\\00d7  "}'
             '#demo-table table{width: 100%; table-layout: fixed}'
             '#demo-table th:nth-child(1){width: 25%}'
             '#demo-table th:nth-child(2){width: 40%}'
             '#demo-table td{vertical-align: top}'
             '</style>')


def tag_links(tags, href=''):
    return ''.join(f'<a class="demo-tag" href="{href}#tag={t}">{t}</a>' for t in tags)


rows, used_tags = [], set()
with open(OUT_DIR / 'demos.rst', 'w') as index:
    print("""
************
Demo Scripts
************

.. toctree::
   :maxdepth: 1
   :hidden:

"""[1:-1], file=index)
    for demo in demos:
        app = import_module('pymordemos.' + demo).app
        app._name = 'pymor-demo ' + demo
        tags = getattr(import_module('pymordemos.' + demo), 'TAGS', ())
        if unknown := set(tags) - set(TAG_VOCABULARY):
            raise ValueError(f'Unknown tags in demo {demo}: {", ".join(sorted(unknown))}')
        used_tags.update(tags)
        description = escape((getdoc(app.default_command) or '').split('\n')[0])
        rows.append(f'<tr data-tags="{" ".join(tags)}"><td><a href="{demo}.html">{demo}</a></td>'
                    f'<td>{description}</td><td>{tag_links(tags)}</td></tr>')
        with open(OUT_DIR / f'{demo}.rst', 'w') as f:
            docs = generate_rst_docs(app, generate_toc=False).splitlines()
            print('\n'.join(docs[:2]), file=f)
            print('=' * len(demo), file=f)
            print(demo, file=f)
            print('=' * len(demo), file=f)
            if tags:
                print(f'\n.. raw:: html\n\n   {TAG_STYLE}<p>{tag_links(tags, "demos.html")}</p>', file=f)
            print('\n'.join(docs[5:]), file=f)
            print(f"""
Source code
-----------

.. py:function:: main()
   :module: pymordemos.{demo}
   :no-index:

   .. literalinclude:: ../../../src/pymordemos/{demo}.py
      :language: python
      :lines: 5-
""", file=f)
        print('   ' + demo, file=index)

    buttons = ''.join(f'<button class="demo-tag" data-tag="{t}">{t}</button>'
                      for t in TAG_VOCABULARY if t in used_tags)
    html = f"""
{TAG_STYLE}
<p>Select tags to filter demos:
<br />
{buttons}
<button class="demo-tag" data-clear>clear</button></p>
<div id="demo-table"><table><thead><tr><th>Demo</th><th>Description</th><th>Tags</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table></div>
<script>
(function() {{
  const active = new Set();
  const buttons = document.querySelectorAll('button.demo-tag[data-tag]');
  const rows = document.querySelectorAll('#demo-table tbody tr');
  function update() {{
    buttons.forEach(b => b.classList.toggle('active', active.has(b.dataset.tag)));
    rows.forEach(r => {{
      const tags = r.dataset.tags.split(' ');
      r.style.display = [...active].every(t => tags.includes(t)) ? '' : 'none';
    }});
  }}
  function setHash() {{
    history.replaceState(null, '', active.size ? '#tag=' + [...active].join(',') : location.pathname);
    update();
  }}
  function readHash() {{
    active.clear();
    const m = location.hash.match(/^#tag=(.+)$/);
    if (m) m[1].split(',').forEach(t => active.add(decodeURIComponent(t)));
    update();
  }}
  buttons.forEach(b => b.addEventListener('click', () => {{
    active.has(b.dataset.tag) ? active.delete(b.dataset.tag) : active.add(b.dataset.tag);
    setHash();
  }}));
  document.querySelector('button.demo-tag[data-clear]').addEventListener('click', () => {{
    active.clear();
    setHash();
  }});
  window.addEventListener('hashchange', readHash);
  readHash();
}})();
</script>
"""
    print('\n.. raw:: html\n', file=index)
    print('\n'.join('   ' + line if line else '' for line in html.splitlines()), file=index)
