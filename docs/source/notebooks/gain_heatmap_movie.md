---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Making an gain pattern movie

```{code-cell} ipython3
# Imports
import numpy as np
import pyant
%matplotlib notebook
```

First we define and instansiate the beam model

```{code-cell} ipython3
beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=50e6,
    I0=10**4.81,
    radius=10.0,
)
```

It is convenient to define a updating function that will modify the beam at each frame

```{code-cell} ipython3
def update(beam, item):
    beam.radius = item
    beam.elevation += 0.25
    return beam
```

The we use the plotting tools to create a movie of the gain heatmap

```{code-cell} ipython3
_, _, _ pyant.plotting.gain_heatmap_movie(
    beam, iterable=np.linspace(10, 23, num=100), beam_update=update)
```

```{code-cell} ipython3

```
