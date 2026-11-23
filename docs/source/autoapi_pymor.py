import re

MANUAL_SKIPS = ('RENDER_FRAGMENT_SHADER',
                'RENDER_VERTEX_SHADER',
                'pymor.bindings.fenics.RestrictedFenicsOperator.linear',
                'pymor.bindings.fenicsx.FenicsxInterpolationOperator.linear',
                'pymor.bindings.fenicsx.FenicsxInterpolationOperator.source',
                'pymor.core.defaults._default_container',
                'pymor.discretizers.builtin.fv.NonlinearReactionOperator.linear',
                'pymor.discretizers.builtin.grids.referenceelements.Line.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Line.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Point.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Point.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Square.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Square.volume',
                'pymor.discretizers.builtin.grids.referenceelements.Triangle.dim',
                'pymor.discretizers.builtin.grids.referenceelements.Triangle.volume',
                'pymor.discretizers.skfem.cg.SKFemBilinearFormOperator.sparse',
                'pymor.discretizers.skfem.cg.SKFemLinearFormOperator.sparse',
                'pymor.discretizers.skfem.cg.BoundaryDirichletFunctional.sparse',
                'pymor.models.iosys.ImpulseFunction.dim_domain',
                'pymor.models.iosys.StepFunction.dim_domain',
                'pymor.operators.block.adjoint_type',
                'pymor.operators.list.LinearComplexifiedListVectorArrayOperatorBase.linear',
                'pymor.parameters.base.Mu.parameters',
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


def setup(app):
    app.connect('autoapi-skip-member', skip)
    return {'parallel_read_safe': True}
