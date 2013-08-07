tests_require = ['tox', 'pytest', 'pytest-cache', 'pytest-capturelog', 'pytest-instafail',
                 'pytest-xdist', 'pytest-cov', 'pytest-flakes', 'pytest-pep8']
install_requires = ['Cython', 'numpy', 'scipy', 'PyContracts', 'PyOpenGL', 'PySide', 'Sphinx',
                    'docopt', 'dogpile.cache' , 'glumpy', 'numpydoc']
pre_setup_requires = ['cython', 'numpy']
setup_requires = pre_setup_requires + ['nose']
install_suggests = ['ipython', 'ipdb', 'matplotlib', 'sympy'] + tests_require

import_names = {'ipython': 'IPython'}

if __name__ == '__main__':
    print(' '.join([i for i in install_requires + install_suggests]))
