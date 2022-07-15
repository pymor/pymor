# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import sys
import os
import slugify
import glob
import sphinx
import sysconfig
from pathlib import Path

# Check Sphinx version
if sphinx.__version__ < "3.4":
    raise RuntimeError("Sphinx 3.4 or newer required")

needs_sphinx = '3.4'
os.environ['PYMOR_WITH_SPHINX'] = '1'
os.environ['PYBIND11_DIR'] = sysconfig.get_path('purelib')

# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

this_dir = Path(__file__).resolve().parent
src_dir = (this_dir / '..' / '..' / 'src').resolve()
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(this_dir))

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.coverage',
              'sphinx.ext.autosummary',
              'sphinx.ext.linkcode',
              'sphinx.ext.intersphinx',
              'pymordocstring',
              'try_on_binder',
              'myst_nb',
              'sphinx.ext.mathjax',
              'sphinx_qt_documentation',
              'autoapi.extension',
              'autoapi_pymor',
              'sphinxcontrib.bibtex',
              ]
# this enables:
# substitutions-with-jinja2, direct-latex-math and definition-lists
# ref: https://myst-parser.readthedocs.io/en/latest/using/syntax-optional.html
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "substitution",
]
myst_url_schemes = ("http", "https", "mailto")
# auto genereated link anchors
myst_heading_anchors = 2
import substitutions # noqa
myst_substitutions = substitutions.myst_substitutions
nb_execute_notebooks = "cache"
nb_execution_timeout = 180
# print tracebacks to stdout
nb_execution_show_tb = True

bibtex_bibfiles = ['bibliography.bib']
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.md': 'myst-nb',
}


# The master toctree document.
master_doc = 'index'

# General substitutions.
project = 'pyMOR'
copyright = 'pyMOR developers and contributors'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
# imports have to be delayed until after sys.path modification
import pymor  # noqa
import autoapi_pymor # noqa
version = pymor.__version__
rst_epilog = substitutions.substitutions

# The full version, including alpha/beta/rc tags.
release = version.split('-')[0]
print(version, release)

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
# unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "literal"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.

on_gitlab_ci = os.environ.get('GITLAB_CI', 'nope') != 'nope'

html_theme = 'sphinx_material'
html_theme_options = {
    'base_url': 'https://gh-docs.pymor.org/',
    'html_minify': False,
    'css_minify': on_gitlab_ci,
    'nav_title': 'Documentation',
    'globaltoc_depth': 5,
    'theme_color': 'indigo',
    'color_primary': 'indigo',
    'color_accent': 'blue',
    'version_dropdown': True,
    'version_json': '/versions.json'
}
# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "%s v%s Manual" % (project, version)

# The name of an image file (within the static path) to place at the top of
# the sidebar.
html_logo = '../../logo/pymor_logo_white.svg'

# The name of an image file to use as favicon.
html_favicon = '../../logo/pymor_favicon.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# all: "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "searchbox.html"]
}
# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {
#    'index': 'indexcontent.html',
# }

# If false, no module index is generated.
html_use_modindex = True

# If true, the reST sources are included in the HTML build as _sources/<name>.
# html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".html").
# html_file_suffix = '.html'

# Hide link to page source.
html_show_sourcelink = False

# Output file base name for HTML help builder.
htmlhelp_basename = 'pymor'

# Pngmath should try to align formulas properly.
pngmath_use_preview = True


# -----------------------------------------------------------------------------
# LaTeX output
# -----------------------------------------------------------------------------

# The paper size ('letter' or 'a4').
# latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
# _stdauthor = 'Written by the NumPy community'
# latex_documents = [
#    ('reference/index', 'numpy-ref.tex', 'NumPy Reference',
#     _stdauthor, 'manual'),
#    ('user/index', 'numpy-user.tex', 'NumPy User Guide',
#     _stdauthor, 'manual'),
# ]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# Additional stuff for the LaTeX preamble.

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = False


# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = glob.glob("generated/*.rst")

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}

# autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']

# PyQt5 inventory is only used internally, actual link targets PySide2
intersphinx_mapping = {'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://numpy.org/doc/stable/', None),
                       'PyQt5': ("https://www.riverbankcomputing.com/static/Docs/PyQt5", None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/', None),
                       'matplotlib': ('https://matplotlib.org/stable/', None),
                       'Sphinx': (' https://www.sphinx-doc.org/en/stable/', None)}

modindex_common_prefix = ['pymor.']


# make intersphinx link to pyside2 docs
qt_documentation = 'PySide2'

branch = os.environ.get('CI_COMMIT_REF_NAME', 'main')
# this must match PYMOR_ROOT/.ci/gitlab/deploy_docs
try_on_binder_branch = branch.replace('github/PUSH_', 'from_fork__')
try_on_binder_slug = os.environ.get('CI_COMMIT_REF_SLUG', slugify.slugify(try_on_binder_branch))


def linkcode_resolve(domain, info):
    if domain == 'py':
        if not info['module']:
            return None
        filename = info['module'].replace('.', '/')
        return f'https://github.com/pymor/pymor/tree/{branch}/src/{filename}.py'
    return None


autoapi_dirs = [src_dir / 'pymor']
autoapi_type = 'python'
# allows incremental build
autoapi_keep_files = True
autoapi_ignore = ['*/pymordemos/minimal_cpp_demo/*']
suppress_warnings = ["autoapi"]
autoapi_template_dir = this_dir / '_templates' / 'autoapi'
autoapi_member_order = "groupwise"
autoapi_options = ["show-inheritance", "members", "undoc-members"]
