# `pyant`

PyAnt is a standardization of implementing radar gain pattern in Python. Nothing more, nothing less.


## Getting started

To install

```bash
   pip install pyant
```
or the nightly build

```bash
   git clone --branch develop git@github.com:danielk333/pyant.git
   cd pyant
   pip install .
```

## Developing

Please refer to the style and contribution guidelines documented in the
[IRF Software Contribution Guide](https://danielk.developer.irf.se/software_contribution_guide/).
Generally external code-contributions are made trough a "Fork-and-pull"
workflow, while internal contributions follow the branching strategy outlined
in the contribution guide.

## Docs

To make the docs, install the development requirements and run `mkdocs`

```bash
   pip install .[develop]
   mkdocs build --site-dir public
```

or run `mkdocs serve` to live view the docs.

## Notebooks

To develop notebooks for documentation in Jupyter-lab, install the above development requirements as well as `jupyter-lab`.

Then run Jupyter-lab in the appropriate folder `./docs/notebooks` using `jupyter-lab` and
open a new plain-text version file using `jupytext`, e.g. a `light` py file or a `MyST` md file.
You can also pair notebooks with the plain-text files which will keep both in synch.

For more information on how to pair notebooks in order to have persistent plain-text versions,
see the [jupytext docs](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html).
