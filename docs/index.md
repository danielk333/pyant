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

To develop notebooks for documentation in Jupyter-lab, install the following development requirements.

Then run notebooks in the appropriate folder `./notebooks` using `jupyter-notebook` and
pair the new notebook with a MyST file.

For more information on how to pair notebooks in order to have persistent plain-text versions,
see the [jupytext docs](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html).
