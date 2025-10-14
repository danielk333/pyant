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

beam = pyant.models.Cassegrain(
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=930e6,
    outer_radius=40.0,
    inner_radius=23.0,
    peak_gain=10**4.81,
)

pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0)
plt.show()
```
