# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

# This file was originally based upon sphinxcontrib-napoleon
# Copyright 2013 Rob Ruana

from collections import deque, defaultdict, OrderedDict
from types import MethodType, FunctionType
import re
import functools

from sphinx.util.inspect import safe_getattr

STRING_TYPE = str

INCLUDE_SPECIAL_WITH_DOC = False
INCLUDE_PRIVATE_WITH_DOC = False

MEMBER_BLACKLIST = ()

FIELD_SECTIONS = ('parameters', 'yields', 'returns', 'raises', 'attributes')
GENERIC_SECTIONS = ('example', 'examples', 'see also', 'defaults')
KNOWN_SECTIONS = FIELD_SECTIONS + GENERIC_SECTIONS

builtin_function_or_method = type(dict.fromkeys)
method_descriptor = type(dict.get)


def table(rows):
    r = ['.. csv-table::', '    :delim: @', '    :widths: 20, 80', '']
    r.extend('    ' + ' @ '.join(r) for r in rows)
    r.append('')
    return r


class PeekIterator(object):
    def __init__(self, *args):
        self._iterable = iter(*args)
        self._cache = deque()
        if len(args) == 2:
            self.sentinel = args[1]
        else:
            self.sentinel = object()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def _fillcache(self, n):
        if not n:
            n = 1
        try:
            while len(self._cache) < n:
                self._cache.append(next(self._iterable))
        except StopIteration:
            while len(self._cache) < n:
                self._cache.append(self.sentinel)

    def has_next(self):
        return self.peek() != self.sentinel

    def next(self, n=None):
        self._fillcache(n)
        if not n:
            if self._cache[0] == self.sentinel:
                raise StopIteration
            if n is None:
                result = self._cache.popleft()
            else:
                result = []
        else:
            if self._cache[n - 1] == self.sentinel:
                raise StopIteration
            result = [self._cache.popleft() for i in range(n)]
        return result

    def peek(self, n=None):
        self._fillcache(n)
        if n is None:
            result = self._cache[0]
        else:
            result = [self._cache[i] for i in range(n)]
        return result


def get_indent(line):
    for i, s in enumerate(line):
        if not s.isspace():
            return i
    return len(line)


def get_min_indent(lines):
    min_indent = None
    for line in lines:
        if line:
            indent = get_indent(line)
            if min_indent is None:
                min_indent = indent
            elif indent < min_indent:
                min_indent = indent
    return min_indent or 0


def indent(lines, n=4):
    return [(' ' * n) + line for line in lines]


def dedent(lines, full=False):
    if full:
        return [line.lstrip() for line in lines]
    else:
        min_indent = get_min_indent(lines)
        return [line[min_indent:] for line in lines]


def is_indented(line, indent=1):
    for i, s in enumerate(line):
        if i >= indent:
            return True
        elif not s.isspace():
            return False
    return False


def parse_docstring(docstring):

    def consume_contiguous():
        lines = []
        while (line_iter.has_next()
               and line_iter.peek()
               and not is_section_header()):
            lines.append(next(line_iter))
        return lines

    def consume_empty():
        lines = []
        line = line_iter.peek()
        while line_iter.has_next() and not line:
            lines.append(next(line_iter))
            line = line_iter.peek()
        return lines

    def consume_to_next_section():
        consume_empty()
        lines = []
        while not is_section_break():
            lines.append(next(line_iter))
        return lines + consume_empty()

    def is_section_header():
        section, underline = line_iter.peek(2)
        section = section.lower()
        if (section in KNOWN_SECTIONS) and isinstance(underline, STRING_TYPE):
            pattern = r'[=\-`:\'"~^_*+#<>]{' + str(len(section)) + r'}$'
            return bool(re.match(pattern, underline))
        return False

    def is_section_break():
        line1, line2 = line_iter.peek(2)
        return (not line_iter.has_next()
                or is_section_header()
                or ['', ''] == [line1, line2])

    if isinstance(docstring, STRING_TYPE):
        docstring = dedent(docstring.splitlines())
    docstring = [s.rstrip().expandtabs(4) for s in docstring]

    if docstring:
        min_indent = get_min_indent(docstring[1:])
        docstring[0] = docstring[0].lstrip()
        docstring[1:] = [l[min_indent:] for l in docstring[1:]]

    line_iter = PeekIterator(docstring)
    sections = []
    non_section_lines = consume_empty()

    while line_iter.has_next():
        if is_section_header():
            if non_section_lines:
                sections.append(('nonsection', non_section_lines))
                non_section_lines = []
            section = line_iter.next().lower()
            next(line_iter)
            sections.append((section, consume_to_next_section()))
        else:
            if not non_section_lines:
                lines = consume_contiguous() + consume_empty()
            else:
                lines = consume_to_next_section()
            non_section_lines.extend(lines)
    if non_section_lines:
        sections.append(('nonsection', non_section_lines))

    return sections


def parse_fields_section(lines):

    def consume_indented_block(indent=1):
        lines = []
        line = line_iter.peek()
        while(line_iter.has_next() and (not line or is_indented(line, indent))):
            lines.append(next(line_iter))
            line = line_iter.peek()
        return lines

    def consume_field(parse_type=True, prefer_type=False):
        line = next(line_iter)
        if parse_type:
            _name, _, _type = line.partition(':')
        else:
            _name, _type = line, ''
        _name, _type = _name.strip(), _type.strip()
        if prefer_type and not _type:
            _type, _name = _name, _type
        indent = get_indent(line)
        _desc = dedent(consume_indented_block(indent + 1))
        return _name, _type, _desc

    line_iter = PeekIterator(lines)

    while line_iter.has_next() and not line_iter.peek():
        next(line_iter)
    fields = []
    while line_iter.has_next():
        _name, _type, _desc = consume_field()
        if _name or _type or _desc:
            fields.append((_name, _type, _desc,))
    return fields


def parse_generic_section(lines):
    return dedent(lines)


def parse_sections(sections):

    parsed_sections = {}
    non_section_lines = []

    for section, lines in sections:
        if section == 'nonsection':
            non_section_lines.extend(lines)
        elif section in parsed_sections:
            raise ValueError('Duplicate section "{}" in docstring'.format(section))
        elif section in FIELD_SECTIONS:
            parsed_sections[section] = parse_fields_section(lines)
        else:
            parsed_sections[section] = parse_generic_section(lines)

    return non_section_lines, parsed_sections


def format_attributes_section(section, lines):
    formatted_lines = []
    for _name, _type, _desc in lines:
        formatted_lines.append('.. attribute:: ' + _name)
        if _type:
            formatted_lines.append('   :annotation: ' + _type)
        if _desc:
            formatted_lines.extend([''] + indent(_desc, 3))
        formatted_lines.append('')
    return formatted_lines


def format_generic_section(section, lines, use_admonition=False):
    if use_admonition:
        header = '.. admonition:: %s' % section
        lines = indent(lines, 3)
    else:
        header = '.. rubric:: %s' % section
    if lines:
        return [header, ''] + lines + ['']
    else:
        return [header, '']


def format_fields_section(section, lines):
    l = ['.. admonition:: ' + section, '']

    def format_name_type(name, type_):
        return name + ' : ' + type_ if type_ else name

    for n, t, d in lines:
        l.extend(indent([format_name_type(n, t)]))
        l.extend(indent(d, 8))
    return l


def inspect_class(obj):
    methods = defaultdict(list)
    attributes = defaultdict(list)
    mro = obj.__mro__

    def get_full_class_name(c):
        if c.__module__ == '__builtin__':
            return c.__name__
        else:
            return c.__module__ + '.' + c.__name__

    def get_class(m):
        for c in mro:
            if m in getattr(c, '__dict__', []):
                return c
        return None

    # this is pretty lame, should do better
    for c in mro:
        if '_sphinx_documented_attributes' not in c.__dict__:
            format_docstring(c, dont_recurse=True)

    # sorted(dir(obj), key=lambda x: '|' + x if x.startswith('_') else x):
    for k in dir(obj):
        try:
            o = safe_getattr(obj, k)
            is_method = isinstance(o, (MethodType, FunctionType, builtin_function_or_method, method_descriptor))
            if k.startswith('_') and not is_method:
                continue
            if k == '__init__':
                continue
            if k.startswith('_') and not k.endswith('__'):
                continue
            if k.startswith('__') and k.endswith('__') and not o.__doc__:
                continue
            class_ = get_class(k)
            assert class_ is not None
            class_name = getattr(class_, '__name__', '')
            if k.startswith('_' + class_name + '__'):
                k = k.split('_' + class_name)[-1]
            if k in MEMBER_BLACKLIST:
                continue
            if is_method:
                documenting_class = class_
                if not o.__doc__:
                    for c in mro:
                        if k in c.__dict__:
                            try:
                                supero = safe_getattr(c, k)
                                if supero.__doc__:
                                    documenting_class = c
                                    break
                            except AttributeError:
                                pass
                methods[class_].append((k, documenting_class))
            else:
                documenting_class = class_
                if k not in obj._sphinx_documented_attributes:
                    for c in mro:
                        if k in c.__dict__.get('_sphinx_documented_attributes', []):
                            documenting_class = c
                            break
                attributes[class_].append((k, documenting_class))
        except AttributeError:
            pass

    key_func = lambda x: '|' + x[0].lower() if x[0].startswith('_') else x[0].lower()
    methods = {k: [':meth:`~{}.{}`'.format(get_full_class_name(c), n) for n, c in sorted(v, key=key_func)]
               for k, v in methods.items()}
    rows = [(':class:`~{}`'.format(get_full_class_name(c)), ', '.join(methods[c]))
            for c in mro if c is not object and c in methods]
    if rows:
        im = ['.. admonition:: Methods', '']
        im.extend(indent(table(rows)))
        im.append('')
    else:
        im = []

    all_attributes = {x[0] for v in attributes.values() for x in v}
    for c in mro:
        for a in c.__dict__.get('_sphinx_documented_attributes', []):
            if a not in all_attributes:
                attributes[c].append((a, c))
    attributes = {k: [':attr:`~{}.{}`'.format(get_full_class_name(c), n) for n, c in sorted(v, key=key_func)]
                  for k, v in attributes.items()}
    rows = [(':class:`~{}`'.format(get_full_class_name(c)), ', '.join(attributes[c]))
            for c in mro if c is not object and c in attributes]
    if rows:
        ia = ['.. admonition:: Attributes', '']
        ia.extend(indent(table(rows)))
        ia.append('')
    else:
        ia = []

    return im, ia


def format_docstring(obj, lines=None, dont_recurse=False):

    if lines is None:
        lines = obj.__doc__ if obj.__doc__ is not None else ''
    non_section_lines, fields = parse_sections(parse_docstring(lines))
    section_formatters = OrderedDict((
        ('parameters', format_fields_section),
        ('yields', format_fields_section),
        ('returns', format_fields_section),
        ('raises', format_fields_section),
        ('example', format_generic_section),
        ('examples', format_generic_section),
        ('see also', format_generic_section),
        ('_methods', None),
        ('_attributes', None),
        ('attributes', format_attributes_section),
        ('defaults', functools.partial(format_generic_section, use_admonition=True))
    ))

    sections = {}
    for section, lines in fields.items():
        sections[section] = section_formatters[section](section.capitalize(), lines)

    if isinstance(obj, type):
        if 'attributes' in fields:
            obj._sphinx_documented_attributes = [n for n, _, _ in fields['attributes']]
        else:
            try:
                obj._sphinx_documented_attributes = []
            except TypeError:
                pass
        if not dont_recurse:
            sections['_methods'], sections['_attributes'] = inspect_class(obj)

    parsed_lines = non_section_lines
    for section in section_formatters:
        parsed_lines.extend(sections.get(section, []))

    return parsed_lines


def _process_docstring(app, what, name, obj, options, lines):
    lines[:] = format_docstring(obj, lines)


def setup(app):
    from sphinx.application import Sphinx
    if not isinstance(app, Sphinx):
        return  # probably called by tests

    app.connect('autodoc-process-docstring', _process_docstring)

    return {'parallel_read_safe': True}
