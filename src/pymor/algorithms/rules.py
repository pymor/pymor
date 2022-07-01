# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from collections.abc import Iterable, Mapping
from collections import OrderedDict
from weakref import WeakValueDictionary

from pymor.core.base import BasicObject, UberMeta, abstractmethod, classinstancemethod
from pymor.core.exceptions import NoMatchingRuleError, RuleNotMatchingError
from pymor.operators.interface import Operator
from pymor.tools.formatsrc import format_source, print_source
from pymor.tools.table import format_table


class rule:
    """Decorator to make a method a rule in a given |RuleTable|.

    The decorated function will become the :attr:`action` to
    perform in case the rule :meth:`matches`.
    Matching conditions are specified by subclassing and
    overriding the :meth:`matches` method.

    If an action is decorated by multiple rules, all these rules
    must match for the action to apply.

    Attributes
    ----------
    action
        Method to call in case the rule matches.
    """

    def __call__(self, action):
        if isinstance(action, rule):
            self.action = action.action
            self.next_rule = action
            self.num_rules = action.num_rules + 1
        else:
            self.action = action
            self.next_rule = None
            self.num_rules = 1
        return self

    @abstractmethod
    def _matches(self, obj):
        """Returns True if given object matches the condition."""
        pass

    def matches(self, obj):
        """Returns True if given object matches the condition."""
        if self._matches(obj):
            if self.next_rule is None:
                return True
            else:
                return self.next_rule.matches(obj)

    condition_description = None
    condition_type = None

    def _ipython_display_(self):
        print_source(self.action)

    def __repr__(self):
        return format_source(self.action)

    @property
    def action_description(self):
        return self.action.__doc__ or self.action.__name__[len('action_'):]

    @property
    def source(self):
        from inspect import getsourcelines
        return ''.join(getsourcelines(self.action)[0])


class match_class_base(rule):

    def __init__(self, *classes):
        super().__init__()
        if not classes:
            raise ValueError('At least one class is required')
        self.classes = classes
        self.condition_description = ', '.join(c.__name__ for c in classes)


class match_class(match_class_base):
    """|rule| that matches when obj is instance of one of the given classes."""

    condition_type = 'CLASS'

    def _matches(self, obj):
        return isinstance(obj, self.classes)


class match_class_all(match_class_base):
    """|rule| that matches when each item of obj is instance of one of the given classes."""

    condition_type = 'ALLCLASSES'

    def _matches(self, obj):
        return all(isinstance(o, self.classes) for o in obj)


class match_class_any(match_class_base):
    """|rule| that matches when any item of obj is instance of one of the given classes."""

    condition_type = 'ANYCLASS'

    def _matches(self, obj):
        return any(isinstance(o, self.classes) for o in obj)


class match_always(rule):
    """|rule| that always matches."""

    condition_type = 'ALWAYS'

    def __init__(self, action):
        self(action)

    def _matches(self, obj):
        return True


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

    def _matches(self, obj):
        return self.condition(obj)


class RuleTableMeta(UberMeta):
    """Meta class for |RuleTable|."""

    def __new__(cls, name, parents, dct):
        assert 'rules' not in dct
        rules = []
        if not {p.__name__ for p in parents} <= {'RuleTable', 'BasicObject'}:
            raise NotImplementedError('Inheritance for RuleTables not implemented yet.')
        for k, v in dct.items():
            if isinstance(v, rule):
                if not k.startswith('action_'):
                    raise ValueError('Rule definition names have to start with "action_"')
                v.name = k
                rules.append(v)
        # note: since Python 3.6, the definition order is preserved in dct,
        # so rules has the right order
        dct['rules'] = rules
        dct['_breakpoint_for_obj'] = WeakValueDictionary()
        dct['_breakpoint_for_name'] = set()

        return super().__new__(cls, name, parents, dct)

    def __repr__(cls):
        return format_rules(cls.rules)

    def __getitem__(cls, idx):
        return cls.rules[idx]

    __str__ = __repr__


class RuleTable(BasicObject, metaclass=RuleTableMeta):
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
        """:noindex:"""
        assert isinstance(rule_, rule)
        self.rules.insert(index, rule_)

    @classinstancemethod
    def append_rule(cls, rule_):
        assert isinstance(rule_, rule)
        cls.rules.append(rule_)

    @append_rule.instancemethod
    def append_rule(self, rule_):
        """:noindex:"""
        assert isinstance(rule_, rule)
        self.rules.append(rule_)

    @classmethod
    def breakpoint_for_obj(cls, obj):
        """Add a conditional breakpoint for given object.

        Break execution in :meth:`~RuleTable.apply`, when being applied
        to a certain object.

        Parameters
        ----------
        obj
            Object for which to add the conditional breakpoint.
        """
        # By using a WeakValueDictionary we ensure that the breakpoint is
        # removed when the object is finalized and does not match a new
        # object with the same id.
        cls._breakpoint_for_obj[id(obj)] = obj

    @classmethod
    def breakpoint_for_name(cls, name):
        """Add a conditional breakpoint for objects of given name.

        Break execution in :meth:`~RuleTable.apply`, when being applied
        to an object with a certain name.

        Parameters
        ----------
        name
            :attr:`~pymor.core.base.BasicObject.name` of the object for which
            to add the conditional breakpoint.
        """
        cls._breakpoint_for_name.add(name)

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
        if id(obj) in self._breakpoint_for_obj or getattr(obj, 'name', None) in self._breakpoint_for_name:
            try:
                breakpoint()
            except NameError:
                import pdb
                pdb.set_trace()

        if self.use_caching and obj in self._cache:
            return self._cache[obj]

        failed_rules = []

        def matching_rules():
            for r in self.rules:
                if r.matches(obj):
                    yield r
                else:
                    failed_rules.append(r)

        for r in matching_rules():
            try:
                result = r.action(self, obj)
                self._cache[obj] = result
                return result
            except RuleNotMatchingError:
                failed_rules.append(r)

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
            if isinstance(c, dict):
                result[child] = {k: self.apply(v) if v is not None else v for k, v in c.items()}
            elif isinstance(c, (list, tuple)):
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
            2. `a` is a dict` and each of its values is either an |Operator| or `None`.
            3. `a` is a `list` or `tuple` and each of its elements is either an |Operator|
               or `None`.
        """
        children = set()
        for k in obj._init_arguments:
            try:
                v = getattr(obj, k)
                if (isinstance(v, Operator)
                        or isinstance(v, dict) and all(isinstance(vv, Operator) or vv is None for vv in v.values())
                        or isinstance(v, (list, tuple)) and all(isinstance(vv, Operator) or vv is None for vv in v)):
                    children.add(k)
            except AttributeError:
                pass
        return children

    def __repr__(self):
        return super().__repr__() + "\n\n" + format_rules(self.rules)


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


def format_rules(rules):
    rows = [['Pos', 'Match Type', 'Condition', 'Action Name / Action Description', 'Stop']]
    for i, r in enumerate(rules):
        for ii in range(r.num_rules):
            rows.append(['' if ii else str(i),
                         r.condition_type,
                         r.condition_description,
                         '' if ii else r.action_description])
            r = r.next_rule
    return format_table(rows)


def print_rules(rules):
    print(format_rules(rules))
