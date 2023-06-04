PyAnt
=====

:Release: |release|
:Date: |today|

PyAnt is a standardization of implementing radar gain pattern in Python. Nothing more, nothing less.


Getting started
-----------------

To install

.. code-block:: bash

   pip install pyant

or the nightly build

.. code-block:: bash

   git clone --branch develop git@github.com:danielk333/pyant.git
   cd pyant
   pip install .


Tutorials
---------

.. toctree::
   :maxdepth: 2

   notebooks/getting_started


Examples
---------

Example gallery of the different modular functionality of the toolbox.

.. toctree::
   :maxdepth: 2

   autogallery/index


API Reference
==============

.. irf_autopackages:: package
   :template: autosummary/module.rst
   :toctree: autosummary
   :exclude: pyant.version

   pyant


Developing
----------

Please refer to the style and contribution guidelines documented in the
[IRF Software Contribution Guide](https://danielk.developer.irf.se/software_contribution_guide/).
Generally external code-contributions are made trough a "Fork-and-pull"
workflow, while internal contributions follow the branching strategy outlined
in the contribution guide.

Docs
~~~~

To make the docs, use the `Makefile` by running

.. code-block:: bash

   make html


Notebooks
~~~~~~~~~

To develop notebooks for documentation in Jupyter-lab, install the following

.. code-block:: bash

   pip install notebook jupytext

Then run notebooks in the appropriate folder `docs/source/notebooks` using `jupyter-notebook` and
pair the new notebook with a MyST file.

For more information on how to pair notebooks in order to have persistent plain-text versions,
see the [jupytext docs](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html).


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
