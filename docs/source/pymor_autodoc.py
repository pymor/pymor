# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import inspect
import pkgutil
from typing import Generic, ParamSpec, TypeVarTuple, get_origin

from sphinx import addnodes
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.inspect import getannotations, isproperty, safe_getattr
from sphinx.util.typing import stringify_annotation

# private members which are documented nevertheless
DOCUMENTED_PRIVATE_MEMBERS = [
    'pymor.operators.interface.Operator._assemble_lincomb',  # extension point
]

# public members which are not documented
UNDOCUMENTED_PUBLIC_MEMBERS = [
    'pymor.auto_load_jupyter_extension',  # internal, called on import
]


def skip(app, what, name, obj, skip, options):
    fullname = f"{getattr(obj, '__module__', None)}.{getattr(obj, '__qualname__', None)}"
    if fullname in DOCUMENTED_PRIVATE_MEMBERS:
        return False
    if fullname in UNDOCUMENTED_PUBLIC_MEMBERS:
        return True
    return None


def _own_members(cls):
    """Methods and attributes documented by autodoc for the given class."""
    methods, attributes = set(), set()
    for name, value in cls.__dict__.items():
        fullname = f'{cls.__module__}.{cls.__qualname__}.{name}'
        if name.startswith('_') and fullname not in DOCUMENTED_PRIVATE_MEMBERS:
            continue
        if fullname in UNDOCUMENTED_PUBLIC_MEMBERS:
            continue
        if isinstance(value, type) and value.__qualname__ == f'{cls.__qualname__}.{name}':  # nested classes
            continue
        # functools.cached_property is a routine according to inspect, but documented as property
        if inspect.isroutine(safe_getattr(cls, name, None)) and not isproperty(value):
            methods.add(name)
        else:
            attributes.add(name)
    # annotation-only attributes (members in cls.__dict__ have been handled above; autodoc
    # ignores annotations of members inherited from a base class)
    attributes.update(name for name in getannotations(cls) if not name.startswith('_') and name not in dir(cls))
    return methods, attributes


def _member_table(title, role, rows):
    if not rows:
        return []
    lines = ['', f'.. admonition:: {title}', '', '   .. list-table::', '      :widths: 25 75', '']
    for cls, names in rows:
        lines.append(f'      * - :class:`~{cls.__module__}.{cls.__qualname__}`')
        lines.append('        - ' + ', '.join(f':{role}:`~{cls.__module__}.{cls.__qualname__}.{name}`'
                                             for name in sorted(names)))
    lines.append('')
    return lines


def add_member_overview(app, what, name, obj, options, lines):
    """Append tables of all methods and attributes, including inherited ones, to class docstrings.

    Members are grouped by the class in the MRO that defines them, and each member links to
    its documentation in that class. Only pyMOR classes are considered.
    """
    # autosummary also emits autodoc-process-docstring (without members option) to
    # extract summary lines
    if what not in ('class', 'exception') or not isinstance(obj, type) or not getattr(options, 'members', None):
        return
    method_rows, attribute_rows = [], []
    seen = set()
    for cls in obj.__mro__:
        if not cls.__module__.startswith('pymor.'):
            continue
        methods, attributes = _own_members(cls)
        methods, attributes = methods - seen, attributes - seen
        seen |= methods | attributes
        if methods:
            method_rows.append((cls, methods))
        if attributes:
            attribute_rows.append((cls, attributes))
    lines.extend(_member_table('Methods', 'meth', method_rows))
    lines.extend(_member_table('Attributes', 'attr', attribute_rows))


def hide_members_from_toc(app, doctree):
    """Do not add class members to the table of contents (only modules, classes and functions)."""
    for node in doctree.findall(addnodes.desc):
        if node.get('domain') == 'py' and node.get('objtype') in ('method', 'attribute', 'property'):
            node['no-contents-entry'] = node['nocontentsentry'] = True


def _type_params(obj):
    # for classes, only consider the type parameters declared by the class itself
    if isinstance(obj, type) and '__type_params__' not in obj.__dict__:
        return ()
    return getattr(obj, '__type_params__', ())


def add_type_params(app, what, name, obj, options, signature, return_annotation):
    """Add PEP 695 type parameters to signatures, which autodoc does not show.

    See https://github.com/sphinx-doc/sphinx/issues/10568.
    """
    def format_param(param):
        if isinstance(param, ParamSpec):
            result = f'**{param.__name__}'
        elif isinstance(param, TypeVarTuple):
            result = f'*{param.__name__}'
        else:
            result = param.__name__
            if param.__bound__ is not None:
                result += f': {stringify_annotation(param.__bound__, 'smart')}'
            elif param.__constraints__:
                result += f': ({", ".join(stringify_annotation(c, 'smart') for c in param.__constraints__)})'
        if param.has_default():
            result += f' = {stringify_annotation(param.__default__, 'smart')}'
        return result

    params = _type_params(obj)
    if signature is None or not params:
        return None
    return f'[{", ".join(format_param(p) for p in params)}]{signature}', return_annotation


def remove_generic_base(app, name, obj, options, bases):
    """Remove the implicit `Generic[...]` base of classes with PEP 695 type parameters."""
    if isinstance(obj, type) and _type_params(obj):
        bases[:] = [b for b in bases if get_origin(b) is not Generic] or [object]


def _type_params_in_scope(node):
    """Names of the PEP 695 type parameters declared by the descriptions enclosing `node`.

    For classes, this includes the type parameters of the base classes, as autodoc
    shows the inherited signature of `__init__` for classes without their own `__init__`.
    """
    names = set()
    while node is not None:
        if isinstance(node, addnodes.desc):
            for sig in (c for c in node.children if isinstance(c, addnodes.desc_signature)):
                for param in sig.findall(addnodes.desc_type_parameter):
                    names.update(n.astext() for n in param.findall(addnodes.desc_sig_name))
                if node.get('objtype') in ('class', 'exception') and sig.get('fullname'):
                    try:
                        cls = pkgutil.resolve_name(f"{sig.get('module')}:{sig['fullname']}")
                    except (ImportError, AttributeError):
                        continue
                    if isinstance(cls, type):
                        names.update(param.__name__ for c in cls.__mro__ for param in getattr(c, '__type_params__', ()))
        node = node.parent
    return names


class RemoveParameterReferences(SphinxPostTransform):
    """Turn type parameter references into plain text.

    Type parameters are not documented objects, so references to them cause warnings or
    even links to unrelated objects of the same name.

    See https://github.com/sphinx-doc/sphinx/issues/14462 and proposed fix in
    https://github.com/sphinx-doc/sphinx/pull/14518.

    """

    default_priority = 5  # before ReferencesResolver

    def run(self, **kwargs):
        for ref in list(self.document.findall(addnodes.pending_xref)):
            # type parameters are rendered as typing.T, as their __module__ is 'typing'
            target = ref['reftarget'].removeprefix('typing.')
            if ref.get('refdomain') == 'py' and target in _type_params_in_scope(ref):
                ref.replace_self(ref.children)


def setup(app):
    app.connect('autodoc-skip-member', skip)
    # run after napoleon has converted the docstring
    app.connect('autodoc-process-docstring', add_member_overview, priority=600)
    # run before the toctree collector (priority 500)
    app.connect('doctree-read', hide_members_from_toc, priority=400)
    app.connect('autodoc-process-signature', add_type_params)
    app.connect('autodoc-process-bases', remove_generic_base)
    app.add_post_transform(RemoveParameterReferences)
    return {'parallel_read_safe': True}
