# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2016 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from collections import Iterable, Mapping, OrderedDict

from pymor.core.exceptions import NoMatchingRuleError, RuleNotMatchingError
from pymor.core.interfaces import BasicInterface, UberMeta, abstractmethod, classinstancemethod
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.table import format_table


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


class RuleTableMeta(UberMeta):
    def __new__(cls, name, parents, dct):
        assert 'rules' not in dct
        rules = []
        if not {p.__name__ for p in parents} <= {'RuleTable', 'BasicInterface'}:
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
        return format_table(rows)

    def __getitem__(cls, idx):
        return cls._rules[idx]

    __str__ = __repr__


class RuleTable(BasicInterface, metaclass=RuleTableMeta):

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
    def map_children(cls, obj, *args, children=None, **kwargs):
        return cls().map_children(obj, *args, children=children, **kwargs)

    @map_children.instancemethod
    def map_children(self, obj, *args, children=None, **kwargs):
        children = children or self.get_children(obj)
        result = {}
        for child in children:
            c = getattr(obj, child)
            if isinstance(c, Mapping):
                result[child] = {k: self.apply(v, *args, **kwargs) if v is not None else v for k, v in c.items()}
            elif isinstance(c, Iterable):
                result[child] = tuple(self.apply(v, *args, **kwargs) if v is not None else v for v in c)
            else:
                result[child] = self.apply(c, *args, **kwargs) if c is not None else c
        return result

    @classinstancemethod
    def apply_children(cls, obj, *args, children=None, **kwargs):
        return cls().apply_children(obj, *args, children=children, **kwargs)

    @apply_children.instancemethod
    def apply_children(self, obj, *args, children=None, **kwargs):
        self.map_children(obj, *args, children=children, **kwargs)

    @classinstancemethod
    def replace_children(cls, obj, *args, children=None, **kwargs):
        return cls().replace_children(obj, *args, children=children, **kwargs)

    @replace_children.instancemethod
    def replace_children(self, obj, *args, children=None, **kwargs):
        return obj.with_(**self.map_children(obj, *args, children=children, **kwargs))

    @classmethod
    def get_children(cls, obj):
        children = set()
        for k in obj.with_arguments:
            try:
                v = getattr(obj, k)
                if (isinstance(v, OperatorInterface) or
                    isinstance(v, Mapping) and all(isinstance(vv, OperatorInterface) or vv is None for vv in v.values()) or
                    isinstance(v, Iterable) and type(v) is not str and all(isinstance(vv, OperatorInterface) or vv is None for vv in v)):
                    children.add(k)
            except AttributeError:
                pass
        return children


def print_children(obj):
    def build_tree(obj):

        def process_child(child):
            c = getattr(obj, child)
            if isinstance(c, Mapping):
                return child, OrderedDict((k + ': ' + v.name, build_tree(v)) for k, v in sorted(c.items()))
            elif isinstance(c, Iterable):
                return child, OrderedDict((str(i) + ': ' + v.name, build_tree(v)) for i, v in enumerate(c))
            else:
                return child + ': ' + c.name, build_tree(c)

        return OrderedDict(process_child(child) for child in sorted(RuleTable.get_children(obj)))

    try:
        from asciitree import LeftAligned
        print(LeftAligned()({obj.name: build_tree(obj)}))
    except ImportError:
        from pprint import pprint
        pprint({obj.name: build_tree(obj)})
