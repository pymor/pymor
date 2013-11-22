# -*- coding: utf-8 -*-
# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

# This file is based upon sphinxcontrib-napoleon
# Copyright 2013 Rob Ruana

from collections import deque, defaultdict, OrderedDict
from types import MethodType, FunctionType
import re

from sphinx.util.inspect import safe_getattr

INCLUDE_SPECIAL_WITH_DOC = False
INCLUDE_PRIVATE_WITH_DOC = False

MEMBER_BLACKLIST = tuple()


def table(rows):
    r = ['.. csv-table::', '    :delim: @', '    :widths: 20, 80', '']
    r.extend('    ' + ' @ '.join(r) for r in rows)
    r.append('')
    return r

class peek_iter(object):
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
                self._cache.append(self._iterable.next())
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


class Docstring(object):

    def __init__(self, docstring, app=None, what='', name='',
                 obj=None, options=None):
        self._app = app
        self._what = what
        self._name = name
        self._obj = obj
        self._opt = options
        if isinstance(docstring, basestring):
            docstring = docstring.splitlines()
        docstring = [s.rstrip() for s in docstring]
        self._line_iter = peek_iter(docstring)
        self._parsed_lines = []
        self._is_in_section = False
        if not hasattr(self, '_sections'):
            self._sections = OrderedDict((
                ('parameters', self._parse_parameters_section),
                ('yields', self._parse_yields_section),
                ('returns', self._parse_returns_section),
                ('raises', self._parse_raises_section),
                ('example', self._parse_generic_section),
                ('examples', self._parse_generic_section),
                ('see also', self._parse_see_also_section),
                ('_methods', None),
                ('_attributes', None),
                ('attributes', self._parse_attributes_section),
            ))
        self._parse()

    def lines(self):
        return self._parsed_lines

    def _consume_indented_block(self, indent=1):
        lines = []
        line = self._line_iter.peek()
        while(not self._is_section_break()
              and (not line or self._is_indented(line, indent))):
            lines.append(self._line_iter.next())
            line = self._line_iter.peek()
        return lines

    def _consume_contiguous(self):
        lines = []
        while (self._line_iter.has_next()
               and self._line_iter.peek()
               and not self._is_section_header()):
            lines.append(self._line_iter.next())
        return lines

    def _consume_empty(self):
        lines = []
        line = self._line_iter.peek()
        while self._line_iter.has_next() and not line:
            lines.append(self._line_iter.next())
            line = self._line_iter.peek()
        return lines

    def _consume_field(self, parse_type=True, prefer_type=False):
        line = self._line_iter.next()
        if parse_type:
            _name, _, _type = line.partition(':')
        else:
            _name, _type = line, ''
        _name, _type = _name.strip(), _type.strip()
        if prefer_type and not _type:
            _type, _name = _name, _type
        indent = self._get_indent(line)
        _desc = self._dedent(self._consume_indented_block(indent + 1))
        _desc = self.__class__(_desc).lines()
        return _name, _type, _desc

    def _consume_fields(self, parse_type=True, prefer_type=False):
        self._consume_empty()
        fields = []
        while not self._is_section_break():
            _name, _type, _desc = self._consume_field(parse_type, prefer_type)
            if _name or _type or _desc:
                fields.append((_name, _type, _desc,))
        return fields

    def _consume_returns_section(self):
        return self._consume_fields(prefer_type=True)

    def _consume_section_header(self):
        section = self._line_iter.next()
        self._line_iter.next()
        return section

    def _consume_to_next_section(self):
        self._consume_empty()
        lines = []
        while not self._is_section_break():
            lines.append(self._line_iter.next())
        return lines + self._consume_empty()

    def _dedent(self, lines, full=False):
        if full:
            return [line.lstrip() for line in lines]
        else:
            min_indent = self._get_min_indent(lines)
            return [line[min_indent:] for line in lines]

    def _format_admonition(self, admonition, lines):
        lines = self._strip_empty(lines)
        if len(lines) == 1:
            return ['.. %s:: %s' % (admonition, lines[0].strip()), '']
        elif lines:
            lines = self._indent(self._dedent(lines), 3)
            return ['.. %s::' % admonition, ''] + lines + ['']
        else:
            return ['.. %s::' % admonition, '']

    def _format_block(self, prefix, lines, padding=None):
        if lines:
            if padding is None:
                padding = ' ' * len(prefix)
            result_lines = []
            for i, line in enumerate(lines):
                if line:
                    if i == 0:
                        result_lines.append(prefix + line)
                    else:
                        result_lines.append(padding + line)
                else:
                    result_lines.append('')
            return result_lines
        else:
            return [prefix]

    def _format_field(self, _name, _type, _desc):
        separator = any([s for s in _desc]) and ' --' or ''
        if _name:
            if _type:
                field = ['**%s** (*%s*)%s' % (_name, _type, separator)]
            else:
                field = ['**%s**%s' % (_name, separator)]
        elif _type:
            field = ['*%s*%s' % (_type, separator)]
        else:
            field = []
        return field + _desc

    def _format_fields(self, field_type, fields):
        field_type = ':%s:' % field_type.strip()
        padding = ' ' * len(field_type)
        multi = len(fields) > 1
        lines = []
        for _name, _type, _desc in fields:
            field = self._format_field(_name, _type, _desc)
            if multi:
                if lines:
                    lines.extend(self._format_block(padding + ' * ', field))
                else:
                    lines.extend(self._format_block(field_type + ' * ', field))
            else:
                lines.extend(self._format_block(field_type + ' ', field))
        return lines

    def _get_current_indent(self, peek_ahead=0):
        line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        while line != self._line_iter.sentinel:
            if line:
                return self._get_indent(line)
            peek_ahead += 1
            line = self._line_iter.peek(peek_ahead + 1)[peek_ahead]
        return 0

    def _get_indent(self, line):
        for i, s in enumerate(line):
            if not s.isspace():
                return i
        return len(line)

    def _get_min_indent(self, lines):
        min_indent = None
        for line in lines:
            if line:
                indent = self._get_indent(line)
                if min_indent is None:
                    min_indent = indent
                elif indent < min_indent:
                    min_indent = indent
        return min_indent or 0

    def _indent(self, lines, n=4):
        return [(' ' * n) + line for line in lines]

    def _is_indented(self, line, indent=1):
        for i, s in enumerate(line):
            if i >= indent:
                return True
            elif not s.isspace():
                return False
        return False

    def _is_section_header(self):
        section, underline = self._line_iter.peek(2)
        section = section.lower()
        if section in self._sections and isinstance(underline, basestring):
            pattern = r'[=\-`:\'"~^_*+#<>]{' + str(len(section)) + r'}$'
            return bool(re.match(pattern, underline))
        return False

    def _is_section_break(self):
        line1, line2 = self._line_iter.peek(2)
        return (not self._line_iter.has_next()
                or self._is_section_header()
                or ['', ''] == [line1, line2])

    def _parse(self):
        non_section_lines = self._consume_empty()
        sections = {}
        while self._line_iter.has_next():
            if self._is_section_header():
                try:
                    section = self._consume_section_header().lower()
                    self._is_in_section = True
                    lines = self._sections[section.lower()](section)
                    sections[section] = lines
                finally:
                    self._is_in_section = False
            else:
                if not non_section_lines:
                    lines = self._consume_contiguous() + self._consume_empty()
                else:
                    lines = self._consume_to_next_section()
                non_section_lines.extend(lines)
        self._parsed_lines = non_section_lines

        if isinstance(self._obj, type):
            self._inspect_class(sections)
        for section in self._sections:
            self._parsed_lines.extend(sections.get(section, []))

    def _inspect_class(self, sections):
        methods = defaultdict(list)
        attributes = defaultdict(list)
        mro = self._obj.__mro__

        def get_full_class_name(c):
            return c.__module__ + '.' + c.__name__

        def get_class(m):
            for c in mro:
                if m in getattr(c, '__dict__', []):
                    return c
            return None

        for k in dir(self._obj):
            try:
                o = safe_getattr(self._obj, k)
                is_method = isinstance(o, (MethodType, FunctionType))
                if k[0] == '_' and not is_method:
                    continue
                class_ = get_class(k)
                assert class_ is not None
                class_name = getattr(class_, '__name__', '')
                full_class_name = get_full_class_name(class_)
                if k.startswith('_' + class_name + '__'):
                    k = k.split('_' + class_name)[-1]
                if k in MEMBER_BLACKLIST:
                    continue
                if is_method:
                    methods[class_].append(':meth:`~{}.{}`'.format(full_class_name, k))
                else:
                    attributes[class_].append(':attr:`~{}.{}`'.format(full_class_name, k))
            except AttributeError:
                pass

        rows = [(':class:`~{}`'.format(get_full_class_name(c)), ', '.join(methods[c]))
                for c in mro if c is not object and methods[c]]
        if rows:
            im = ['.. admonition:: Methods', '']
            im.extend(self._indent(table(rows)))
            im.append('')
            sections['_methods'] = im

        rows = [(':attr:`~{}`'.format(get_full_class_name(c)), ', '.join(attributes[c]))
                for c in mro if c is not object and attributes[c]]
        if rows:
            ia = ['.. admonition:: Attributes', '']
            ia.extend(self._indent(table(rows)))
            ia.append('')
            sections['_attributes'] = ia

    def _parse_attributes_section(self, section):
        lines = []
        for _name, _type, _desc in self._consume_fields():
            lines.append('.. attribute:: ' + _name)
            if _type:
                lines.append('   :annotation: ' + _type)
            if _desc:
                lines.extend([''] + self._indent(_desc, 3))
            lines.append('')
        return lines

    def _parse_generic_section(self, section, use_admonition=False):
        lines = self._strip_empty(self._consume_to_next_section())
        lines = self._dedent(lines)
        if use_admonition:
            header = '.. admonition:: %s' % section
            lines = self._indent(lines, 3)
        else:
            header = '.. rubric:: %s' % section
        if lines:
            return [header, ''] + lines + ['']
        else:
            return [header, '']

    def _parse_parameters_section(self, section):
        fields = self._consume_fields()
        l = ['.. admonition:: Parameters', '']

        def format_name_type(name, type_):
            return name + ' : ' + type_ if type_ else name

        l.extend(self._indent(table([(format_name_type(name, type_), ' '.join(descr))
                                     for name, type_, descr in fields])))
        return l

    def _parse_raises_section(self, section):
        fields = self._consume_fields()
        field_type = ':raises:'
        padding = ' ' * len(field_type)
        multi = len(fields) > 1
        lines = []
        for _name, _type, _desc in fields:
            sep = _desc and ' -- ' or ''
            if _name:
                if ' ' in _name:
                    _name = '**%s**' % _name
                else:
                    _name = ':exc:`%s`' % _name
                if _type:
                    field = ['%s (*%s*)%s' % (_name, _type, sep)]
                else:
                    field = ['%s%s' % (_name, sep)]
            elif _type:
                field = ['*%s*%s' % (_type, sep)]
            else:
                field = []
            field = field + _desc
            if multi:
                if lines:
                    lines.extend(self._format_block(padding + ' * ', field))
                else:
                    lines.extend(self._format_block(field_type + ' * ', field))
            else:
                lines.extend(self._format_block(field_type + ' ', field))
        return lines

    def _parse_returns_section(self, section):
        fields = self._consume_returns_section()
        multi = len(fields) > 1

        lines = []
        for _name, _type, _desc in fields:
            field = self._format_field(_name, _type, _desc)
            if multi:
                if lines:
                    lines.extend(self._format_block('          * ', field))
                else:
                    lines.extend(self._format_block(':returns: * ', field))
            else:
                lines.extend(self._format_block(':returns: ', field))
        return lines

    def _parse_see_also_section(self, section):
        lines = self._consume_to_next_section()
        return self._format_admonition('seealso', lines)

    def _parse_yields_section(self, section):
        fields = self._consume_fields(prefer_type=True)
        return self._format_fields('Yields', fields)

    def _strip_empty(self, lines):
        if lines:
            start = -1
            for i, line in enumerate(lines):
                if line:
                    start = i
                    break
            if start == -1:
                lines = []
            end = -1
            for i in reversed(xrange(len(lines))):
                line = lines[i]
                if line:
                    end = i
                    break
            if start > 0 or end + 1 < len(lines):
                lines = lines[start:end + 1]
        return lines


def _process_docstring(app, what, name, obj, options, lines):
    lines[:] = Docstring(lines, app, what, name, obj, options).lines()


def _skip_member(app, what, name, obj, skip, options):
    has_doc = getattr(obj, '__doc__', False)
    is_member = (what == 'class' or what == 'exception' or what == 'module')
    if name != '__weakref__' and name != '__init__' and has_doc and is_member:
        cls = getattr(obj, 'im_class', getattr(obj, '__objclass__', None))
        cls_is_owner = (cls and hasattr(cls, name) and name in cls.__dict__)
        if what == 'module' or cls_is_owner:
            is_special = name.startswith('__') and name.endswith('__')
            is_private = not is_special and name.startswith('_')
            if (is_special and INCLUDE_SPECIAL_WITH_DOC) or (is_private and INCLUDE_PRIVATE_WITH_DOC):
                return False
    return skip


def setup(app):
    from sphinx.application import Sphinx
    if not isinstance(app, Sphinx):
        return  # probably called by tests

    app.connect('autodoc-process-docstring', _process_docstring)
    app.connect('autodoc-skip-member', _skip_member)
