import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'SAR'
copyright = '2022, Hesham Mostafa'
author = 'Hesham Mostafa'

# The full version, including alpha/beta/rc tags
release = '1.0'
import sar  # noqa

add_module_names = False

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',

]

def autodoc_skip_member_handler(app, what, name, obj, skip, options):
    return name == 'forward' or name == 'extra_repr'

#def setup(app):
#    app.connect('autodoc-skip-member', autodoc_skip_member_handler)

templates_path = ['_templates']


html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
