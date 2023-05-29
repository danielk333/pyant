# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import warnings
from datetime import date
import pyant

# -- Project information -----------------------------------------------------

project = "pyant"
version = ".".join(pyant.__version__.split(".")[:2])
release = pyant.__version__

copyright = f"2020-{date.today().year}, Daniel Kastinen"
author = "Daniel Kastinen"


# -- General configuration ---------------------------------------------------

add_module_names = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
    "myst_nb",
    "sphinx_gallery.load_style",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["*.ipynb", "examples"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".myst": "myst-nb",
}

# The master toctree document.
master_doc = "index"

# MyST-NB config
myst_enable_extensions = [
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "basic"
html_css_files = [
    "https://www.irf.se/branding/irf.css",
    "https://www.irf.se/branding/irf-sphinx-basic.css",
]
html_favicon = "static/favicon.png"
html_logo = "static/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["static"]


# -- Options for gallery extension ---------------------------------------
sphinx_gallery_conf = {
    "examples_dirs": "../../examples",  # path to your example scripts
    "gallery_dirs": "autogallery",  # path where to save gallery generated examples
    "filename_pattern": r".*\.py",
    "ignore_pattern": r".*__no_agl\.py",
}

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Matplotlib is currently using agg, which is a"
    " non-GUI backend, so cannot show the figure.",
)

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False


# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
}
