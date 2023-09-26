#!/usr/bin/env python3

import re
import sys

import nbformat

notebook = nbformat.read(sys.argv[1], nbformat.NO_CONVERT)

for cell in notebook.cells:
    if cell['cell_type'] != 'markdown':
        continue
    md = cell['source']

    md = re.sub(r'```{try_on_binder}.*?```',
                r'',
                md, 0, re.DOTALL)

    md = re.sub(r'{(mod|class|func|meth|attr)}`~[^`]*\.([^`]*)`',
                r'`\2`',
                md)
    md = re.sub(r'{(mod|class|func|meth|attr)}`([^`]*?)\s*<[^`]*>\s*`',
                r'`\2`',
                md)
    md = re.sub(r'{{\s*(.*?)\s*}}',
                r'`\1`',
                md)

    cell['source'] = md

nbformat.write(notebook, sys.argv[1])
