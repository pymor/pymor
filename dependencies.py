tests_require = ['mock', 'nose-cov', 'nose', 'nosehtmloutput', 'nose-progressive', 'tissue']
install_requires = ['distribute', 'numpy', 'scipy', 'PyContracts', 'sympy'
                    'docopt', 'dogpile.cache' , 'numpydoc']
pre_setup_requires = ['cython', 'numpy']
setup_requires = pre_setup_requires + ['nose']
install_suggests = ['matplotlib', 'sympy'] + tests_require

import_names = {'nose-progressive': 'noseprogressive', 'nose-cov': 'nose_cov',
                'nosehtmloutput': 'htmloutput'}
