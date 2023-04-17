project(
    'minimal_cpp_demo',
    'cpp',
    version: '0.1',
    default_options: ['cpp_std=c++14']
)

pymod = import('python')
py3_inst = pymod.find_installation('python3')

pybind11 = dependency('pybind11')

py3_inst.extension_module(
    'model',
    sources: ['model.cc'],
    dependencies: [pybind11],
    override_options: ['cpp_rtti=true']
)
