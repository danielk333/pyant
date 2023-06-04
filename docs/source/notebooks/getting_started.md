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

```{code-cell} ipython3
# Import the used libraries
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
%matplotlib inline

import pyant
```

# Getting started

This is a quick tutorial going trough the basic functionality of the `Beam` class models.

+++

## Instansiate and use a model

We will start by creating a `Airy` disk model, a good first order approximation of a dish-antenna that is used to model a perfect focusing lens with a circular aperture.

```{code-cell} ipython3
beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)
```

Here we instantiated the model by setting the operating frequency, the peak gain, the aperture radius and the initial pointing direction (the additional keyword argument `degrees=True` tells the constructor that angles are given in degrees).

To calculate a gain in a direction we can use two different methods, `sph_gain` and `gain` that take a direction as input, either in spherical (hence the `sph`) coordinates or cartesian coordinates.

```{code-cell} ipython3
az, el = (45.0, 85.0)
k = pyant.coordinates.sph_to_cart(np.array([az, el, 1]), degrees=True)
print(f"az={az} deg, el={el} deg")
print(f"k_x={k[0]:.3f}, k_y={k[1]:.3f}, k_z={k[2]:.3f}")

G = beam.sph_gain(az, el)
print(f"beam.sph_gain(az, el) = {G}")
G = beam.gain(k)
print(f"beam.gain(k) = {G}")
```

We can also re-point the beam by using `sph_point` and `point` respectively

```{code-cell} ipython3
beam.sph_point(45.0, 86.0)
G = beam.sph_gain(az, el)
print(f"pointed beam.sph_gain(az, el) = {G}")

# Note here that `k` should technically be a unit vector but
# a normalization is applied inside the function
beam.point(np.array([1, 1, 1]))
G = beam.gain(k)
print(f"pointed beam.gain(k) = {G}")
```

We can also look at the current pointing and parameters of the beam trough a few convenience properties by default but always trough the `get_parameters` method.

```{code-cell} ipython3
print(f"beam.shape: {beam.shape}")
print(f"beam.keys : {beam.keys}")

# If we request `named` a dict will be returned instead of a list
# otherwise, parameters are always ordered in the same order as `keys`
params, shape = beam.get_parameters(named=True)
for key, val in params.items():
    print(f"{key}:\n{val}\n")
```

We can also vectorize the input direction for which we get the radiation pattern

```{code-cell} ipython3
beam.sph_point(0, 90)
k = np.array([[0, 0, 1.0], [0, 1, 1]]).T

G = beam.gain(k)
print(f"vectored beam.gain(k) = {G}")
```

## Vectorized parameters

Parameters need not be scalar values, this is supported to reduce necessary overhead in setting parameter values between iterations or improve computation time if vectorization should happen across other variables.

We can set parameters three ways:
1. directly trough the `parameters` attribute
2. through the conveince method `fill_parameter`
3. trough a `@property` handle that has been set specifically for that parameter

These behave a bit differently, too show a few examples

```{code-cell} ipython3
# The beam expcets all parameters to be numpy arrays
beam.parameters["frequency"] = np.array([930e6, 230e6])
print(f"\nbeam.shape: {beam.shape}")
pprint(beam.parameters)

# Given a scalar, will fill the current array
beam.fill_parameter("frequency", 930e6)
print(f"\nbeam.shape: {beam.shape}")
pprint(beam.parameters)

# Given a iterable, will replace the current array
beam.fill_parameter("frequency", [930e6, 230e6])
print(f"\nbeam.shape: {beam.shape}")
pprint(beam.parameters)

# There are standard properties for
# pointing, azimuth, elevation, frequency and wavelength
beam.pointing = k
print(f"\nbeam.shape: {beam.shape}")
pprint(beam.parameters)
```

Or we can simply set vectors at initialization

```{code-cell} ipython3
beam = pyant.models.Airy(
    azimuth=[0, 45.0, 0],
    elevation=[90.0, 80.0, 60.0],
    frequency=[930e6, 230e6],
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)
```

We can use the `ind` argument to index our parameters in several ways

```{code-cell} ipython3
G = beam.gain(k[:, 0], ind=(0, 0, 0))
print(f"beam.gain(k[:, 0], ind=(0, 0, 0)) = {G}")

G = beam.gain(k[:, 0], ind=dict(pointing=0, frequency=0, radius=0))
print(f"beam.gain(k[:, 0], ind=dict(pointing=0, frequency=0, radius=0)) = {G}")
```

The `ind` parameter supports any indexing operation that a numpy array can handle. When fetching in dict mode missing keys are treated as `slice(None)`. Also, the default `ind` is `slice(None` for all parameters.

```{code-cell} ipython3
params, _ = beam.get_parameters(ind=(slice(None), 1, 0))
pprint(params)

params, _ = beam.get_parameters(ind={"pointing": 0})
pprint(params)
```

The dimensions of the returned gain reflect the indexing that was made on the parameters and the input wave vector, if a single direction vector and integer indexing is given, a scalar will be returned while if a slice of a parameter is given (even if that parameter has a single value) the returned gain will be an array of size 1.

```{code-cell} ipython3
G = beam.gain(k[:, 0], ind=dict(pointing=0, frequency=0, radius=0))
print(f"beam.gain(k[:, 0], ind=dict(pointing=0, frequency=0, radius=0)) = {G}")

G = beam.gain(k[:, 0], ind=dict(pointing=0, frequency=0, radius=slice(None)))
print(f"beam.gain(k[:, 0], ind=dict(pointing=0, frequency=0, radius=0)) = {G}")
```

An important note is:
**Most models only allow one of these parameters to be vectorized at the time**

This is to reduce the code complexity and allow for less messy optimizations, if there is a dire need to have multiple vectorizations simultaneously, this can be implemented on a case by case basis as a new subclass that re-implements the gain function.

```{code-cell} ipython3
print(f"The traceback with k.shape={k.shape} and all parameters as vectors\n")
try:
    G = beam.gain(k)
except AssertionError as e:
    print(e)
```

However, the input direction vector can be vectorized together with one other component, where the first axis is the parameter shape and the second axis is the input wave vector shape, for example

```{code-cell} ipython3
G = beam.gain(k, ind=(slice(None), 0, 0))
print(f"G = {G}")
print(f"G.shape = {G.shape}")

G = beam.gain(k, ind=(1, slice(None), 0))
print(f"G = {G}")
print(f"G.shape = {G.shape}")

G = beam.gain(k, ind=(1, 0, 0))
print(f"G = {G}")
print(f"G.shape = {G.shape}")
```

## Plotting

It is very often we need to illustrate the radiation pattern, to support this there is a plotting module included with several standard visualization functions

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=80)

plot_list = [
    (0, 0, axes[0, 0]),
    (0, 1, axes[0, 1]),
    (2, 0, axes[1, 0]),
    (2, 1, axes[1, 1]),
]

for i, j, ax in plot_list:
    pyant.plotting.gain_heatmap(
        beam,
        resolution=301,
        min_elevation=80.0,
        ax=ax,
        ind={
            "pointing": i,
            "frequency": j,
        },
    )
    pstr = f"az={beam.azimuth[i]:.1f} deg | el={beam.elevation[i]:.1f} deg"
    ax.set_title(f"{pstr} | f:{beam.frequency[j]/1e6:.1f} MHz", fontsize=14)
```
