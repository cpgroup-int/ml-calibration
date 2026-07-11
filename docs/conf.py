"""Sphinx configuration for the MADMAX closed-loop calibration docs."""

import madmax_calibration

# -- Project information -----------------------------------------------------

project = "MADMAX Closed-Loop Calibration"
author = "MADMAX calibration project"
release = madmax_calibration.__version__
version = release
copyright = "2026, MADMAX calibration project"

# -- General configuration ---------------------------------------------------

# Note: sphinx.ext.intersphinx is deliberately omitted so the docs build
# fully offline (no external objects.inv fetches).
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# MyST (Markdown) settings.
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 4

# Autodoc settings.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"MADMAX Closed-Loop Calibration {release}"
html_static_path = []

# -- Design-document math conversion ----------------------------------------
#
# The original design documents (docs/design/*.md) are kept byte-identical
# to the source documents and use LaTeX \[ ... \] / \( ... \) math
# delimiters.  MyST's dollarmath extension expects $$ ... $$ / $ ... $, so
# the delimiters are converted at build time; the committed files are not
# modified.


def _convert_design_math(app, docname, source):
    if docname.startswith("design/"):
        src = source[0]
        src = src.replace("\\[", "$$").replace("\\]", "$$")
        src = src.replace("\\(", "$").replace("\\)", "$")
        source[0] = src


def setup(app):
    app.connect("source-read", _convert_design_math)
