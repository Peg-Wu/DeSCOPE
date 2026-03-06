# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import pathlib

sys.path.insert(0, (pathlib.Path(__file__).parents[2] / "src").resolve().as_posix())

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'descope'
copyright = '2026, Pengpeng Wu'
author = 'Pengpeng Wu'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]

templates_path = ['_templates']
exclude_patterns = []

autoclass_content = "both"
add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_show_sphinx = False
html_static_path = ['_static']
html_logo = "_static/logo.png"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "logo_only": True,
}
html_css_files = [
    "css/custom.css",
]
html_show_sourcelink = False
