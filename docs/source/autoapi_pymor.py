import re

MANUAL_SKIPS = ('RENDER_FRAGMENT_SHADER',
                'RENDER_VERTEX_SHADER',
                'pymor.analyticalproblems.domaindescriptions.DomainDescription.boundary_types',
                'pymor.analyticalproblems.domaindescriptions.DomainDescription.dim',
                'pymor.analyticalproblems.domaindescriptions.PolygonalDomain.boundary_types',
                'pymor.analyticalproblems.expressions.Expression.shape',
                'pymor.analyticalproblems.instationary.InstationaryProblem.name',
                'pymor.bindings.fenics.RestrictedFenicsOperator.linear',
                'pymor.bindings.fenics.RestrictedFenicsOperator.solver_options',
                'pymor.bindings.fenics.RestrictedFenicsOperator.H',
                'pymor.core.base.BasicObject.logger',
                'pymor.core.base.BasicObject.logging_disabled',
                'pymor.core.base.BasicObject.name',
                'pymor.core.base.BasicObject.uid',
                'pymor.core.cache.CacheRegion.persistent',
                'pymor.core.cache.DiskRegion.persistent',
                'pymor.core.cache.MemoryRegion.persistent',
                'pymor.core.cache.CacheableObject.cache_id',
                'pymor.core.cache.CacheableObject.cache_region',
                'pymor.core.defaults._default_container',
                'pymor.discretizers.builtin.fv.NonlinearReactionOperator.linear',
                'pymor.discretizers.builtin.fv.NonlinearReactionOperator.solver_options',
                'pymor.discretizers.builtin.fv.NonlinearReactionOperator.H',
                'pymor.discretizers.builtin.gui.jupyter.threejs.ColorBarRenderer.trait_defaults',
                'pymor.discretizers.builtin.gui.jupyter.threejs.ColorBarRenderer.trait_values',
                'pymor.discretizers.builtin.gui.jupyter.threejs.ColorBarRenderer.get_state',
                'pymor.discretizers.builtin.gui.jupyter.threejs.Renderer.trait_defaults',
                'pymor.discretizers.builtin.gui.jupyter.threejs.Renderer.trait_values',
                'pymor.discretizers.builtin.gui.jupyter.threejs.Renderer.get_state',
                'pymor.discretizers.builtin.gui.jupyter.threejs.ThreeJSPlot.trait_defaults',
                'pymor.discretizers.builtin.gui.jupyter.threejs.ThreeJSPlot.trait_values',
                'pymor.discretizers.builtin.gui.jupyter.threejs.ThreeJSPlot.get_state',
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
                'pymor.discretizers.skfem.cg.AdvectionOperator.sparse',
                'pymor.discretizers.skfem.cg.DiffusionOperator.sparse',
                'pymor.discretizers.skfem.cg.L2Functional.sparse',
                'pymor.discretizers.skfem.cg.L2ProductOperator.sparse',
                'pymor.models.interface.Model.dim_output',
                'pymor.models.interface.Model.linear',
                'pymor.models.interface.Model.order',
                'pymor.models.interface.Model.products',
                'pymor.models.interface.Model.solution_space',
                'pymor.models.iosys.BilinearModel.dim_output',
                'pymor.models.iosys.BilinearModel.order',
                'pymor.models.iosys.LTIModel.dim_output',
                'pymor.models.iosys.LTIModel.order',
                'pymor.models.iosys.LinearDelayModel.dim_output',
                'pymor.models.iosys.LinearDelayModel.order',
                'pymor.models.iosys.LinearStochasticModel.dim_output',
                'pymor.models.iosys.LinearStochasticModel.order',
                'pymor.models.iosys.PHLTIModel.dim_output',
                'pymor.models.iosys.PHLTIModel.order',
                'pymor.models.iosys.SecondOrderModel.dim_output',
                'pymor.models.iosys.SecondOrderModel.order',
                'pymor.models.neural_network.FullyConnectedNN.to',
                'pymor.models.neural_network.FullyConnectedNN.apply',
                'pymor.models.neural_network.FullyConnectedNN.add_module',
                'pymor.models.neural_network.FullyConnectedNN.register_buffer',
                'pymor.models.neural_network.FullyConnectedNN.register_parameter',
                'pymor.models.neural_network.FullyConnectedNN.named_parameters',
                'pymor.models.neural_network.FullyConnectedNN.named_buffers',
                'pymor.parallel.ipython.RemoteId.to_bytes',
                'pymor.tools.mpi.ObjectId.to_bytes',
                'pymor.vectorarrays.mpi.RegisteredLocalSpace.to_bytes',
                'pymor.parallel.basic.GenericRemoteObject.removed',
                'pymor.operators.block.BlockOperatorBase.H',
                'pymor.operators.block.BlockOperatorBase.linear',
                'pymor.operators.block.BlockOperatorBase.solver_options',
                'pymor.operators.block.adjoint_type',
                'pymor.operators.interface.Operator.H',
                'pymor.operators.interface.Operator.solver_options',
                'pymor.operators.list.ListVectorArrayOperatorBase.H',
                'pymor.operators.list.ListVectorArrayOperatorBase.linear',
                'pymor.operators.list.ListVectorArrayOperatorBase.solver_options',
                'pymor.operators.list.LinearComplexifiedListVectorArrayOperatorBase.linear',
                'pymor.operators.list.LinearComplexifiedListVectorArrayOperatorBase.solver_options',
                'pymor.operators.list.LinearComplexifiedListVectorArrayOperatorBase.H',
                'pymor.operators.numpy.NumpyMatrixBasedOperator.sparse',
                'pymor.parallel.interface.RemoteObject.removed',
                'pymor.parallel.dummy.DummyRemoteObject.removed',
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
                'pymor.vectorarrays.interface.VectorSpace.id',
                'pymor.vectorarrays.interface.VectorSpace.is_scalar',
                'pymor.discretizers.builtin.gui.matplotlib.Matplotlib1DWidget.draw_idle',
                'pymor.discretizers.builtin.gui.matplotlib.Matplotlib1DWidget.get_default_filetype',
                'pymor.discretizers.builtin.gui.matplotlib.Matplotlib1DWidget.print_figure',
                'pymor.discretizers.builtin.gui.matplotlib.Matplotlib1DWidget.print_jpg',
                'pymor.discretizers.builtin.gui.matplotlib.MatplotlibPatchWidget.draw_idle',
                'pymor.discretizers.builtin.gui.matplotlib.MatplotlibPatchWidget.get_default_filetype',
                'pymor.discretizers.builtin.gui.matplotlib.MatplotlibPatchWidget.print_figure',
                'pymor.discretizers.builtin.gui.matplotlib.MatplotlibPatchWidget.print_jpg',
                )

SKIPS_RE = re.compile(r'(?:{})'.format('|'.join(map(re.escape, sorted(MANUAL_SKIPS, key=len, reverse=True)))))


def skip(app, what, name, obj, skip, options):
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
