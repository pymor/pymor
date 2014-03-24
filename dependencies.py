tests_require = ['pytest', 'pytest-cache', 'pytest-capturelog']
install_requires = ['Cython', 'numpy', 'scipy', 'Sphinx', 'docopt', 'dogpile.cache']
pre_setup_requires = ['cython', 'numpy']
setup_requires = pre_setup_requires
install_suggests = ['ipython', 'ipdb', 'matplotlib', 'PyOpenGL', 'PySide', 'glumpy'] + tests_require
#install_suggests = ['ipython', 'ipdb', 'matplotlib', 'pyvtk', 'sympy', 'PyOpenGL', 'PySide', 'glumpy'] + tests_require

import_names = {'ipython': 'IPython',
                'pytest-cache': 'pytest_cache',
                'pytest-capturelog': 'pytest_capturelog',
                'pytest-instafail': 'pytest_instafail',
                'pytest-xdist': 'xdist',
                'pytest-cov': 'pytest_cov',
                'pytest-flakes': 'pytest_flakes',
                'pytest-pep8': 'pytest_pep8',
                'PyOpenGL': 'OpenGL'}

if __name__ == '__main__':
    print(' '.join([i for i in install_requires + install_suggests]))
