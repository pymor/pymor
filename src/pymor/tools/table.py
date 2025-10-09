# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from itertools import zip_longest
from textwrap import wrap

import numpy as np


def _wrap_entry(entry, width):
    # simple hack to make wrap break lines on '.' instead of '-'
    # this is mainly for print_defaults
    if '°' in entry:
        raise ValueError
    entry = entry.replace('-', '°').replace('.', '-')
    lines = wrap(entry, width, subsequent_indent='  ', break_on_hyphens=True)
    lines = [l.replace('-', '.').replace('°', '-') for l in lines]
    return lines


def format_table(rows, width='AUTO', title=None):
    rows = [[str(c) for c in r] for r in rows]
    if width == 'AUTO':
        try:
            from shutil import get_terminal_size
            width = get_terminal_size()[0] - 1
        except ImportError:
            width = 1000000
    rows = list(rows)
    column_widths = [max(map(len, c)) for c in zip(*rows, strict=True)]
    if sum(column_widths) + 2*(len(column_widths) - 1) > width:
        largest_column = np.argmax(column_widths)
        column_widths[largest_column] = 0
        min_width = max(column_widths)
        column_widths[largest_column] = max(min_width, width  - 2*(len(column_widths) - 1) - sum(column_widths))
    total_width = sum(column_widths) + 2*(len(column_widths) - 1)

    wrapped_rows = []
    for row in rows:
        cols = [_wrap_entry(c, width=cw) for c, cw in zip(row, column_widths, strict=True)]
        for r in zip_longest(*cols, fillvalue=''):
            wrapped_rows.append(r)
    rows = wrapped_rows

    rows.insert(1, ['-' * cw for cw in column_widths])

    if title is not None:
        separator = '=' * len(title)
        title = (f'{title:^{total_width}}\n'
                 f'{separator:^{total_width}}\n\n')
    else:
        title = ''

    return title + '\n'.join('  '.join(f'{c:<{cw}}' for c, cw in zip(r, column_widths, strict=True))
                             for r in rows)
