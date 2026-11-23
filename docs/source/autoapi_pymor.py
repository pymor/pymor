import re

from sphinx import addnodes
from sphinx.transforms.post_transforms import SphinxPostTransform

MANUAL_SKIPS = ('RENDER_FRAGMENT_SHADER',
                'RENDER_VERTEX_SHADER',
                'pymor.analyticalproblems.domaindescriptions.DomainDescription.boundary_types',
                'pymor.analyticalproblems.domaindescriptions.DomainDescription.dim',
                'pymor.analyticalproblems.expressions.Expression.shape',
                'pymor.bindings.fenics.RestrictedFenicsOperator.linear',
                'pymor.bindings.fenicsx.FenicsxInterpolationOperator.linear',
                'pymor.bindings.fenicsx.FenicsxInterpolationOperator.source',
                'pymor.core.base.BasicObject.logger',
                'pymor.core.base.BasicObject.logging_disabled',
                'pymor.core.base.BasicObject.name',
                'pymor.core.base.BasicObject.uid',
                'pymor.core.cache.CacheRegion.persistent',
                'pymor.core.cache.CacheableObject.cache_id',
                'pymor.core.cache.CacheableObject.cache_region',
                'pymor.core.defaults._default_container',
                'pymor.discretizers.builtin.fv.NonlinearReactionOperator.linear',
                'pymor.discretizers.builtin.grids.interfaces.BoundaryInfo.boundary_types',
                'pymor.discretizers.builtin.grids.interfaces.ReferenceElement.dim',
                'pymor.discretizers.builtin.grids.interfaces.ReferenceElement.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Line.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Line.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Point.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Point.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Square.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Square.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Triangle.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Triangle.volume',
                'pymor.discretizers.builtin.grids.subgrid.SubGrid.parent_grid',
                'pymor.discretizers.skfem.cg.SKFemBilinearFormOperator.sparse',
                'pymor.discretizers.skfem.cg.SKFemLinearFormOperator.sparse',
                'pymor.discretizers.skfem.cg.BoundaryDirichletFunctional.sparse',
                'pymor.models.interface.Model.dim_output',
                'pymor.models.interface.Model.linear',
                'pymor.models.interface.Model.order',
                'pymor.models.interface.Model.products',
                'pymor.models.interface.Model.solution_space',
                'pymor.models.iosys.ImpulseFunction.dim_domain',
                'pymor.models.iosys.StepFunction.dim_domain',
                'pymor.operators.block.adjoint_type',
                'pymor.operators.list.LinearComplexifiedListVectorArrayOperatorBase.linear',
                'pymor.operators.numpy.NumpyMatrixBasedOperator.sparse',
                'pymor.parallel.interface.RemoteObject.removed',
                'pymor.parameters.base.Mu.parameters',
                'pymor.parameters.base.ParametricObject.parameters',
                'pymor.parameters.base.ParametricObject.parameters_inherited',
                'pymor.parameters.base.ParametricObject.parameters_internal',
                'pymor.parameters.base.ParametricObject.parameters_own',
                'pymor.parameters.base.ParametricObject.parametric',
                'pymor.vectorarrays.interface.VectorArray.base',
                'pymor.vectorarrays.interface.VectorArray.dim',
                'pymor.vectorarrays.interface.VectorArray.ind',
                'pymor.vectorarrays.interface.VectorArray.is_view',
                'pymor.vectorarrays.interface.VectorSpace.dim',
                'pymor.vectorarrays.interface.VectorSpace.is_scalar',
                )

SKIPS_RE = re.compile(r'(?:{})'.format('|'.join(map(re.escape, sorted(MANUAL_SKIPS, key=len, reverse=True)))))


def skip(app, what, name, obj, skip, options):
    if name == 'pymor.operators.interface.Operator._assemble_lincomb':
        return False
    try:
        if ':noindex:' in obj.docstring:
            print(f'HERE DO_SKIP {name}')
            return True
    except AttributeError:
        pass
    return SKIPS_RE.search(name)


def _class_type_params(objects, name, seen=None):
    """Type parameters of the class `name` and of all its base classes."""
    seen = set() if seen is None else seen
    obj = objects.get(name.split('[', 1)[0])
    if obj is None or obj.id in seen or obj.type != 'class':
        return set()
    seen.add(obj.id)
    names = {param[1] for param in obj.obj.get('type_params', ())}
    module = obj.id.rsplit('.', 1)[0]
    for base in obj.bases:
        # autoapi does not qualify bases defined in the same module if they are decorated.
        # imported bases are always qualified, so an unqualified base refers to this module
        if base.split('[', 1)[0] not in objects:
            base = f'{module}.{base}'
        names |= _class_type_params(objects, base, seen)
    return names


def _type_params_in_scope(node, objects):
    """Names of the PEP 695 type parameters declared by the descriptions enclosing `node`.

    For classes, this includes the type parameters of the base classes, as autoapi
    shows the inherited signature of `__init__` for classes without their own `__init__`.
    """
    names = set()
    while node is not None:
        if isinstance(node, addnodes.desc):
            for sig in (c for c in node.children if isinstance(c, addnodes.desc_signature)):
                for param in sig.findall(addnodes.desc_type_parameter):
                    names.update(n.astext() for n in param.findall(addnodes.desc_sig_name))
                if node.get('objtype') == 'class' and sig.get('fullname'):
                    # recursively look up class type parameters
                    names |= _class_type_params(objects, f"{sig['module']}.{sig['fullname']}")
        node = node.parent
    return names


class PythonTypeReferences(SphinxPostTransform):
    """Make type parameter references plain text and expand `np` to `numpy`.

    Type parameters are not documented objects, so references to them cause warnings or
    even links to unrelated objects of the same name.

    See https://github.com/sphinx-doc/sphinx/issues/14462 and proposed fix in
    https://github.com/sphinx-doc/sphinx/pull/14518.

    `np.` reference targets are expanded to `numpy.`, as autoapi does not resolve module aliases in
    the values of type aliases.
    """

    default_priority = 5  # before ReferencesResolver

    def run(self, **kwargs):
        objects = getattr(self.env, 'autoapi_all_objects', {})
        for ref in list(self.document.findall(addnodes.pending_xref)):
            if ref.get('refdomain') != 'py':
                continue
            target = ref['reftarget']
            if target.startswith('np.'):
                ref['reftarget'] = 'numpy.' + target[3:]
            elif target in _type_params_in_scope(ref, objects):
                ref.replace_self(ref.children)


def setup(app):
    app.connect('autoapi-skip-member', skip)
    app.add_post_transform(PythonTypeReferences)
    return {'parallel_read_safe': True}
