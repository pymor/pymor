project(
    'minimal_cpp_demo',
    'cpp',
    version: '0.1',
    default_options: ['cpp_std=c++14']
)

pymod = import('python')
py3_inst = pymod.find_installation('python3')

pybind11_config = find_program('pybind11-config')
pybind11_config_ret = run_command(pybind11_config, ['--includes'], check: true)
pybind11 = declare_dependency(
    include_directories: [pybind11_config_ret.stdout().split('-I')[-1].strip()],
)

python3 = dependency(py3_inst.path())

py3avlw = py3_inst.extension_module(
    'model',
    sources: ['model.cc'],
    dependencies: [python3, pybind11],
    link_language: 'cpp',
    override_options: ['cpp_rtti=true']
)
