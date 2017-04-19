# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.exceptions import NoMatchingRuleError, RuleNotMatchingError
from pymor.core.interfaces import classinstancemethod


class rule():
    _rules_created = [0]

    def __init__(self, condition, condition_description=None):
        self.condition = \
            condition if isinstance(condition, tuple) else (condition,) if isinstance(condition, type) else condition
        self.condition_description = condition_description
        self._rule_nr = self._rules_created[0]
        self._rules_created[0] += 1

    def __call__(self, action):
        self.action = action
        return self

    def matches(self, obj):
        return isinstance(obj, self.condition) if isinstance(self.condition, tuple) else self.condition(obj)

    def __repr__(self):
        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import Terminal256Formatter
            return highlight(self.source, PythonLexer(), Terminal256Formatter(style='native'))
        except ImportError:
            return self.source

    @property
    def source(self):
        from inspect import getsourcefile, getsourcelines
        with open(getsourcefile(self.action), 'rt') as f:
            source = f.readlines()
        start_line = getsourcelines(self.action)[1] - 1
        lines = [source[start_line].lstrip()]
        indent = len(source[start_line]) - len(lines[0])
        seen_def = False
        for l in source[start_line + 1:]:
            if not seen_def and l.lstrip().startswith('def'):
                seen_def = True
                lines.append(l[indent:])
                continue
            if 0 < len(l) - len(l.lstrip()) <= indent:
                break
            lines.append(l[indent:])
        return ''.join(lines)


class RuleTableMeta(type):
    def __new__(cls, name, parents, dct):
        assert 'rules' not in dct
        rules = []
        for k, v in dct.items():
            if isinstance(v, rule):
                v.name = k
                rules.append(v)
        rules = tuple(sorted(rules, key=lambda r: r._rule_nr))
        dct['_rules'] = rules

        return super().__new__(cls, name, parents, dct)

    def __repr__(cls):
        rows = [['Pos', 'Type', 'Condition Description', 'Action Description']]
        for i, r in enumerate(cls._rules):
            rows.append([str(i),
                         'CLASS' if isinstance(r.condition, tuple) else 'GENERIC',
                         ', '.join(c.__name__ for c in r.condition) if isinstance(r.condition, tuple) else
                         (r.condition_description or str(r.condition)),
                         r.action.__doc__ or 'n.a.'])
        column_widths = [max(map(len, c)) for c in zip(*rows)]
        rows.insert(1, ['-' * cw for cw in column_widths])
        return '\n'.join('  '.join('{:<{}}'.format(c, cw) for c, cw in zip(r, column_widths))
                         for r in rows)

    def __getitem__(cls, idx):
        return cls._rules[idx]

    __str__ = __repr__


class RuleTable(metaclass=RuleTableMeta):

    def __init__(self):
        self._cache = {}

    @classinstancemethod
    def apply(cls, obj, *args, **kwargs):
        return cls().apply(obj, *args, **kwargs)

    @apply.instancemethod
    def apply(self, obj, *args, **kwargs):
        if obj in self._cache:
            return self._cache[obj]
        for r in self._rules:
            if r.matches(obj):
                try:
                    result = r.action(self, obj, *args, **kwargs)
                    self._cache[obj] = result
                    return result
                except RuleNotMatchingError:
                    pass

        raise NoMatchingRuleError('No rule could be applied to {}'.format(obj))
