# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.exceptions import NoMatchingRuleError, RuleNotMatchingError
from pymor.core.interfaces import abstractmethod, classinstancemethod


class rule():
    _rules_created = [0]

    def __init__(self):
        self._rule_nr = self._rules_created[0]
        self._rules_created[0] += 1

    def __call__(self, action):
        self.action = action
        return self

    @abstractmethod
    def matches(self, obj):
        pass

    condition_description = None
    condition_type = None

    def __repr__(self):
        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import Terminal256Formatter
            return highlight(self.source, PythonLexer(), Terminal256Formatter(style='native'))
        except ImportError:
            return self.source

    @property
    def action_description(self):
        return self.action.__doc__ or self.action.__name__[len('action_'):]

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


class match_class(rule):

    condition_type = 'CLASS'

    def __init__(self, *classes):
        super().__init__()
        if not classes:
            raise ValueError('At least one class is required')
        self.classes = classes
        self.condition_description = ', '.join(c.__name__ for c in classes)

    def matches(self, obj):
        return isinstance(obj, self.classes)


class match_generic(rule):

    condition_type = 'GENERIC'

    def __init__(self, condition, condition_description=None):
        super().__init__()
        self.condition = condition
        self.condition_description = condition_description or 'n.a.'

    def matches(self, obj):
        return self.condition(obj)


class RuleTableMeta(type):
    def __new__(cls, name, parents, dct):
        assert 'rules' not in dct
        rules = []
        if not {p.__name__ for p in parents} <= {'RuleTable'}:
            raise NotImplementedError('Inheritance for RuleTables not implemented yet.')
        for k, v in dct.items():
            if isinstance(v, rule):
                if not k.startswith('action_'):
                    raise ValueError('Rule definition names have to start with "action_"')
                v.name = k
                rules.append(v)
        rules = tuple(sorted(rules, key=lambda r: r._rule_nr))
        dct['_rules'] = rules

        return super().__new__(cls, name, parents, dct)

    def __repr__(cls):
        rows = [['Pos', 'Match Type', 'Condition', 'Action Name / Action Description', 'Stop']]
        for i, r in enumerate(cls._rules):
            rows.append([str(i),
                         r.condition_type,
                         r.condition_description,
                         r.action_description])
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

    @classinstancemethod
    def apply_children(cls, obj, *args, **kwargs):
        return cls().apply_children(obj, *args, **kwargs)

    @apply_children.instancemethod
    def apply_children(self, obj, *args, **kwargs):
        return obj.map_children(self.apply, *args, return_new_object=False, **kwargs)

    @classinstancemethod
    def replace_children(cls, obj, *args, **kwargs):
        return cls().replace_children(obj, *args, **kwargs)

    @replace_children.instancemethod
    def replace_children(self, obj, *args, **kwargs):
        return obj.map_children(self.apply, *args, return_new_object=True, **kwargs)
