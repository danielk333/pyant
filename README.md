# PyAnt

PyAnt is a standardization for implementing radar gain pattern in Python. Nothing more, nothing less.

## Getting started

To install

```bash
    pip install pyant
```

or for the nightly build

```bash
    git clone git@github.com:danielk333/pyant.git
    cd pyant
    git checkout develop
    pip install .
```

Alternatively, if you are following updates closely you can install using ``pip install -e .`` so that in the future a ``git pull`` will update the library.

Then get started by looking at the examples gallery and API in the Documentation.

## Example

Plot the gain pattern of a dish radar modeled using the Cassegrain model.

```python
import matplotlib.pyplot as plt
import pyant

ant = pyant.Cassegrain(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    a0=23.0,
    a1=40.0,
    degrees=True,
)

pyant.plotting.gain_heatmap(ant, resolution=301, min_elevation=80.0)
plt.show()
```
