#!/usr/bin/env python
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

#make sure we got distribute in place
from distribute_setup import use_setuptools
use_setuptools()

import sys
import os
import subprocess
from setuptools import find_packages
from distutils.extension import Extension

_orig_generate_a_pyrex_source = None

def _numpy_monkey():
    '''Apparently we need to monkey numpy's distutils to be able to build
    .pyx with Cython instead of Pyrex. The monkeying below is copied from
    https://github.com/matthew-brett/du-cy-numpy/blob/master/matthew_monkey.py
    via the discussion at http://comments.gmane.org/gmane.comp.python.numeric.general/37752
    '''
    global _build_src
    from os.path import join as pjoin, dirname
    from distutils.dep_util import newer_group
    from distutils.errors import DistutilsError

    from numpy.distutils.misc_util import appendpath
    from numpy.distutils import log

    from numpy.distutils.command import build_src
    _orig_generate_a_pyrex_source = build_src.build_src.generate_a_pyrex_source

    def generate_a_pyrex_source(self, base, ext_name, source, extension):
        ''' Monkey patch for numpy build_src.build_src method

        Uses Cython instead of Pyrex, iff source contains 'pymor'

        Assumes Cython is present
        '''
        if not 'pymor' in source:
            return _orig_generate_a_pyrex_source(self, base, ext_name, source, extension)
        
        if self.inplace:
            target_dir = dirname(base)
        else:
            target_dir = appendpath(self.build_src, dirname(base))
        target_file = pjoin(target_dir, ext_name + '.c')
        depends = [source] + extension.depends
        if self.force or newer_group(depends, target_file, 'newer'):
            import Cython.Compiler.Main
            log.info("cythonc:> %s" % (target_file))
            self.mkpath(target_dir)
            options = Cython.Compiler.Main.CompilationOptions(
                defaults=Cython.Compiler.Main.default_options,
                include_path=extension.include_dirs,
                output_file=target_file)
            cython_result = Cython.Compiler.Main.compile(source,
                                                    options=options)
            if cython_result.num_errors != 0:
                raise DistutilsError("%d errors while compiling %r with Cython" \
                    % (cython_result.num_errors, source))
        return target_file


    
    build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source

def write_version():
    revstring = subprocess.check_output(['git', 'describe', '--tags', '--candidates', '20', '--match', '*.*.*']).strip()
    filename = os.path.join(os.path.dirname(__file__), 'src', 'pymor', 'version.py')
    with open(filename, 'w') as out:
        out.write('revstring = \'{}\''.format(revstring))


def _setup(**kwargs):
    '''we'll make use of Distribution's __init__ downloading setup_requires packages right away here'''
    from setuptools.dist import Distribution
    dist = Distribution(kwargs)
    
    # now that we supposedly have at least numpy + cython installed, use them
    # they're dropped in cwd as egg-dirs however. let's discover those first
    from pkg_resources import require
    require("Cython")
    require("numpy")
    _numpy_monkey()
    import Cython.Distutils
    cmdclass = {'build_ext': Cython.Distutils.build_ext}
    #from numpy import get_include
    ext_modules = [Extension("pymor.tools.relations", ["src/pymor/tools/relations.pyx"])]#, include_dirs=[get_include()])]
    kwargs['cmdclass'] = cmdclass
    kwargs['ext_modules'] = ext_modules
 
    # lastly we'll need to tweak matplotlibs config dir or else 
    # installing it from setup will result in a SandboxViolation
    import os
    os.environ['MPLCONFIGDIR'] = "."
 
    from numpy.distutils.core import setup
    setup(**kwargs)


def setup_package():
    write_version()

    tests_requires = ['mock', 'nose-cov', 'nose', 'nosehtmloutput', 'nose-progressive', 'tissue>=0.8']
    install_requires = ['distribute', 'scipy', 'numpy', 'PyContracts',
                        'docopt', 'dogpile.cache' , 'numpydoc'] + tests_requires
    setup_requires = ['cython', 'numpy']
        
    _setup(
        name='pyMor',
        version='0.1.0',
        author='pyMor developers',
        author_email='pymor-dev@listserv.uni-muenster.de',
        maintainer='Rene Milk',
        maintainer_email='rene.milk@wwu.de',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        scripts=['bin/%s' % n for n in [] ] + ['run_tests.py', 'distribute_setup.py'],
        url='http://pymor.org',
        description=' ' ,
        long_description=open('README.txt').read(),
        setup_requires=setup_requires,
        tests_require=tests_requires,
        install_requires=install_requires,
        classifiers=['Development Status :: 2 - Pre-Alpha',
            'License :: OSI Approved :: BSD License',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Visualization'],
        license = 'LICENSE.txt',
    )

if __name__ == '__main__':
    setup_package()
