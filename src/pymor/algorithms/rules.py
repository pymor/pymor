# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2018 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from collections import Iterable, Mapping, OrderedDict

from pymor.core.exceptions import NoMatchingRuleError, RuleNotMatchingError
from pymor.core.interfaces import BasicInterface, UberMeta, abstractmethod, classinstancemethod
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.table import format_table


class rule():
    """Decorator to make a method a rule in a given |RuleTable|.

    The decorated function will become the :attr:`action` to
    perform in case the rule :meth:`matches`.
    Matching conditions are specified by subclassing and
    overriding the :meth:`matches` method.

    Attributes
    ----------
    action
        Method to call in case the rule matches.
    """
    _rules_created = [0]

    def __init__(self):
        self._rule_nr = self._rules_created[0]
        self._rules_created[0] += 1

    def __call__(self, action):
        self.action = action
        return self

    @abstractmethod
    def matches(self, obj):
        """Returns True if given object matches the condition."""
        pass

    condition_description = None
    condition_type = None

    def __repr__(self):
        try:
            from pygments import highlight
            from pygments.lexers import PythonLexer
            from pygments.formatters import Terminal256Formatter
            return highlight(self.source, PythonLexer(), Terminal256Formatter())
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
    """|rule| that matches when obj is instance of one of the given classes."""

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
    """|rule| with matching condition given by an arbitrary function.

    Parameters
    ----------
    condition
        Function of one argument which checks if given object
        matches condition.
    condition_description
        Optional string describing the condition implemented by
        `condition`.
    """

    condition_type = 'GENERIC'

    def __init__(self, condition, condition_description=None):
        super().__init__()
        self.condition = condition
        self.condition_description = condition_description or 'n.a.'

    def matches(self, obj):
        return self.condition(obj)


class RuleTableMeta(UberMeta):
    """Meta class for |RuleTable|."""
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
        rules = list(sorted(rules, key=lambda r: r._rule_nr))
        dct['rules'] = rules

        return super().__new__(cls, name, parents, dct)

    def __repr__(cls):
        rows = [['Pos', 'Match Type', 'Condition', 'Action Name / Action Description', 'Stop']]
        for i, r in enumerate(cls.rules):
            rows.append([str(i),
                         r.condition_type,
                         r.condition_description,
                         r.action_description])
        return format_table(rows)

    def __getitem__(cls, idx):
        return cls.rules[idx]

    __str__ = __repr__


class RuleTable(BasicInterface, metaclass=RuleTableMeta):
    """Define algorithm by a table of match conditions and corresponding actions.

    |RuleTable| manages a table of |rules|, stored in the `rules`
    attributes, which can be :meth:`applied <apply>` to given
    objects.

    A new table is created by subclassing |RuleTable| and defining
    new methods which are decorated with :class:`match_class`,
    :class:`match_generic` or another :class:`rule` subclass.
    The order of the method definitions determines the order in
    which the defined |rules| are applied.

    Parameters
    ----------
    use_caching
        If `True`, cache results of :meth:`apply`.

    Attributes
    ----------
    rules
        `list` of all defined |rules|.
    """

    def __init__(self, use_caching=False):
        self.use_caching = use_caching
        self._cache = {}
        self.rules = list(self.rules)  # make a copy of the list of rules

    @classinstancemethod
    def insert_rule(cls, index, rule_):
        assert isinstance(rule_, rule)
        cls.rules.insert(index, rule_)

    @insert_rule.instancemethod
    def insert_rule(self, index, rule_):
        assert isinstance(rule_, rule)
        self.rules.insert(index, rule_)

    @classinstancemethod
    def append_rule(cls, rule_):
        assert isinstance(rule_, rule)
        cls.rules.append(rule_)

    @append_rule.instancemethod
    def append_rule(self, rule_):
        assert isinstance(rule_, rule)
        self.rules.append(rule_)

    def apply(self, obj):
        """Sequentially apply rules to given object.

        This method iterates over all rules of the given |RuleTable|.
        For each |rule|, it is checked if it :meth:`~rule.matches` the given
        object. If `False`, the next |rule| in the table is considered.
        If `True` the corresponding :attr:`~rule.action` is executed with
        `obj` as parameter. If execution of :attr:`~action` raises
        :class:`~pymor.core.exceptions.RuleNotMatchingError`, the rule is
        considered as not matching, and execution continues with evaluation
        of the next rule. Otherwise, execution is stopped and the return value
        of :attr:`rule.action` is returned to the caller.

        If no |rule| matches, a :class:`~pymor.core.exceptions.NoMatchingRuleError`
        is raised.

        Parameters
        ----------
        obj
            The object to apply the |RuleTable| to.

        Returns
        -------
        Return value of the action of the first matching |rule| in the table.

        Raises
        ------
        NoMatchingRuleError
            No |rule| could be applied to the given object.
        """
        if self.use_caching and obj in self._cache:
            return self._cache[obj]

        for r in self.rules:
            if r.matches(obj):
                try:
                    result = r.action(self, obj)
                    self._cache[obj] = result
                    return result
                except RuleNotMatchingError:
                    pass

        raise NoMatchingRuleError(obj)

    def apply_children(self, obj, children=None):
        """Apply rules to all children of the given object.

        This method calls :meth:`apply` to each child of
        the given object. The children of the object are either provided
        by the `children` parameter or automatically inferred by the
        :meth:`get_children` method.

        Parameters
        ----------
        obj
            The object to apply the |RuleTable| to.
        children
            `None` or a list of attribute names defining the children
            to consider.

        Returns
        -------
        Result of :meth:`apply` for all given children.
        """
        children = children or self.get_children(obj)
        result = {}
        for child in children:
            c = getattr(obj, child)
            if isinstance(c, Mapping):
                result[child] = {k: self.apply(v) if v is not None else v for k, v in c.items()}
            elif isinstance(c, Iterable):
                result[child] = tuple(self.apply(v) if v is not None else v for v in c)
            else:
                result[child] = self.apply(c) if c is not None else c
        return result

    def replace_children(self, obj, children=None):
        """Replace children of object according to rule table.

        Same as :meth:`apply_children`, but additionally calls
        `obj.with_` to replace the children of `obj` with the
        result of the corresponding :meth:`apply` call.
        """
        return obj.with_(**self.apply_children(obj, children=children))

    @classmethod
    def get_children(cls, obj):
        """Determine children of given object.

        This method returns a list of the names of all
        attributes `a`, for which one of the folling is true:

            1. `a` is an |Operator|.
            2. `a` is a `mapping` and each of its values is either an |Operator| or `None`.
            3. `a` is an `iterable` and each of its elements is either an |Operator| or `None`.
        """
        children = set()
        for k in obj.with_arguments:
            try:
                v = getattr(obj, k)
                if (isinstance(v, OperatorInterface) or
                    isinstance(v, Mapping) and all(isinstance(vv, OperatorInterface) or
                                                   vv is None for vv in v.values()) or
                    isinstance(v, Iterable) and type(v) is not str and all(isinstance(vv, OperatorInterface) or
                                                                           vv is None for vv in v)):
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
