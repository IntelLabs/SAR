# Minimal makefile for Sphinx documentation
#

SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	sphinx-build -M help "$(SOURCEDIR)" "$(BUILDDIR)"  $(O)

.PHONY: help Makefile

%: Makefile
	sphinx-build -M  $@ "$(SOURCEDIR)" "$(BUILDDIR)"  $(O) 
