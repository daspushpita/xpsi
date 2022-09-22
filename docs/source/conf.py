# -*- coding: utf-8 -*-
#
# XPSI documentation build configuration file, created by
# sphinx-quickstart on Tue Feb 13 10:59:02 2018.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import print_function

import os
import sys

print('In source directory: ', os.getcwd())

sys.path.insert(0, os.path.abspath('source/'))
sys.path.insert(0, os.path.abspath('../'))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# tested with v1.8.5, but compatibility with other versions unknown
needs_sphinx = '1.8.5'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
                'sphinx.ext.intersphinx',
                'sphinx.ext.viewcode',
                'sphinx.ext.todo',
                'sphinx.ext.coverage',
                'sphinx.ext.mathjax',
                'sphinx.ext.githubpages',
                'sphinx.ext.autosummary',
                'nbsphinx']

intersphinx_mapping = {'sphinx': ('http://www.sphinx-doc.org/en/master', None),
           'numpy': ('https://docs.scipy.org/doc/numpy', None),
           'cython': ('https://cython.readthedocs.io/en/latest', None),
           'emcee': ('https://emcee.readthedocs.io/en/latest', None),
           'getdist': ('https://getdist.readthedocs.io/en/latest', None),
           'schwimmbad': ('https://schwimmbad.readthedocs.io/en/latest', None),
           'mpi4py': ('https://mpi4py.readthedocs.io/en/stable', None),
           'h5py': ('http://docs.h5py.org/en/latest', None),
           'nestcheck': ('https://nestcheck.readthedocs.io/en/latest', None),
           'fgivenx': ('https://fgivenx.readthedocs.io/en/latest', None)}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst','.md']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'x-psi'
copyright = u'2019-2022 the X-PSI Core Team'
author = u'X-PSI Core Team'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u'1.0'
# The full version, including alpha/beta/rc tags.
release = u'1.0.0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['**.ipynb_checkpoints', 'old']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

#autoclass_content = 'both'

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_logo = "./images/xpsilogo_small.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'xpsidoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'xpsi.tex', u'X-PSI Documentation',
     u'Thomas E. Riley', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'xpsi', u'X-PSI Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'X-PSI', u'X-PSI Documentation',
     author, 'X-PSI', 'One line description of project.',
     'Miscellaneous'),
]
