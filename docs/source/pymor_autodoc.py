# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
from typing import Generic, ParamSpec, TypeVarTuple, get_origin

from sphinx import addnodes
from sphinx.transforms.post_transforms import SphinxPostTransform
from sphinx.util.typing import stringify_annotation


def skip(app, what, name, obj, skip, options):
    # private, but documented extension point
    if getattr(obj, '__qualname__', None) == 'Operator._assemble_lincomb':
        return False
    return None


def hide_members_from_toc(app, doctree):
    """Do not add class members to the table of contents (only modules, classes and functions)."""
    for node in doctree.findall(addnodes.desc):
        if node.get('domain') == 'py' and node.get('objtype') in ('method', 'attribute', 'property'):
            node['no-contents-entry'] = node['nocontentsentry'] = True


def _format_type_param(param):
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


def _type_params(obj):
    # for classes, only consider the type parameters declared by the class itself
    if isinstance(obj, type) and '__type_params__' not in obj.__dict__:
        return ()
    return getattr(obj, '__type_params__', ())


def add_type_params(app, what, name, obj, options, signature, return_annotation):
    """Add PEP 695 type parameters to signatures, which autodoc does not show."""
    params = _type_params(obj)
    if signature is None or not params:
        return None
    return f'[{", ".join(_format_type_param(p) for p in params)}]{signature}', return_annotation


def remove_generic_base(app, name, obj, options, bases):
    """Remove the implicit `Generic[...]` base of classes with PEP 695 type parameters."""
    if isinstance(obj, type) and _type_params(obj):
        bases[:] = [b for b in bases if get_origin(b) is not Generic] or [object]


def _class_type_params(module, fullname):
    """Names of the type parameters of the given class and of all its base classes."""
    obj = sys.modules.get(module)
    for part in fullname.split('.'):
        obj = getattr(obj, part, None)
    if not isinstance(obj, type):
        return set()
    return {param.__name__ for cls in obj.__mro__ for param in getattr(cls, '__type_params__', ())}


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
                    names |= _class_type_params(sig.get('module'), sig['fullname'])
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
    # run before the toctree collector (priority 500)
    app.connect('doctree-read', hide_members_from_toc, priority=400)
    app.connect('autodoc-process-signature', add_type_params)
    app.connect('autodoc-process-bases', remove_generic_base)
    app.add_post_transform(RemoveParameterReferences)
    return {'parallel_read_safe': True}
