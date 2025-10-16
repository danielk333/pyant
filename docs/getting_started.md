---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.6
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
    pointing=np.array([0, 0, 1], dtype=np.float64),
    frequency=930e6,
    peak_gain=10**4.81,
    radius=23.0,
)
```

Here we instantiated the model by setting the operating frequency, the peak gain, the aperture radius and the initial pointing direction (given as a 3D pointing vector of unit length, currently pointing in zenith).

To calculate a gain in a direction we can use two different methods, `sph_gain` and `gain` that take a direction as input, either in spherical (hence the `sph`) coordinates or Cartesian coordinates.

```{code-cell} ipython3
az, el = (45.0, 85.0)
k = pyant.coordinates.sph_to_cart(np.array([az, el, 1]), degrees=True)
print(f"az={az} deg, el={el} deg")
print(f"k_x={k[0]:.3f}, k_y={k[1]:.3f}, k_z={k[2]:.3f}")

G = beam.sph_gain(az, el, degrees=True)
print(f"beam.sph_gain(az, el) = {G}")
G = beam.gain(k)
print(f"beam.gain(k) = {G}")
```

We can also re-point the beam by using `sph_point` and `point` respectively

```{code-cell} ipython3
beam.sph_point(45.0, 86.0, degrees=True)
G = beam.sph_gain(az, el, degrees=True)
print(f"pointed beam.sph_gain(az, el) = {G}")
```

We can also look at the current pointing of the beam trough the parameters attribute. The `size` and
`keys` properties tells us the vectorized size of the parameters and which parameters are available.

```{code-cell} ipython3
print(f"beam.size: {beam.size}")
print(f"beam.keys: {beam.keys}")

for key, val in beam.parameters.items():
    print(f"{key}:\n{type(val)}\n")
```

We can also vectorize the input direction for which we get the radiation pattern

```{code-cell} ipython3
beam.sph_point(0, 90, degrees=True)

# Note: the Airy beam can handle non-normalized input wave-vectors, it ignores the length
k_two = np.array([[0, 0, 1.0], [0, 1, 1]]).T

G = beam.gain(k_two)
print(f"vectored beam.gain(k_two) = {G}")
```

## Vectorized parameters

Parameters need not be scalar values, this is supported to reduce necessary overhead in setting
parameter values between iterations or improve computation time if vectorization should happen
across other variables.

There are four possible ways of broadcasting input arrays over the parameters.

### Vectorized `k`, vectorized parameters

The additional axis of all parameters lines up with the input wave vectors additional
axis size, in which case each input k-vector gets evaluated versus each set of
parameters. Input size (3, n), parameter shapes (..., n), output size (n,).

### Vectorized `k`, single parameters

The parameters are all scalars and the input k-vector gets evaluated over this single set.
Input size (3, n), parameter shapes (...), output size (n,).

### Single `k`, vectorized parameters

The additional axis of all parameters line up and the input k-vector is a single vector,
in which case this vector gets computed for all sets of parameters.
Input size (3,), parameter shapes (..., n), output size (n,).

### Single `k`, single parameters

The parameters are all scalars and the input k-vector is a single vector.
Input size (3,), parameter shapes (...), output size ().

### Example

These ways allow for any set of computations and broadcasts (although they need to be prepared
outside the scope of this class) to be set-up using the Beam interface.
For example

```{code-cell} ipython3
# The beam expcets all parameters to be numpy arrays of the same last shape
beam.parameters["pointing"] = np.broadcast_to(
    pyant.coordinates.sph_to_cart(np.array([0, 80.0, 1]), degrees=True).reshape(3, 1),
    (3, 5),
)
beam.parameters["frequency"] = np.linspace(200e6, 930e6, num=5)
beam.parameters["radius"] = np.full((5,), 23.0, dtype=np.float64)

print(f"\nbeam.size: {beam.size}")
G = beam.gain(k)
print(f"beam.gain(k) = {G}")

```
Or we can simply set vectors at initialization. For more in depth examples, see the examples
gallery.

## Plotting

It is very often we need to illustrate the radiation pattern, to support this there is a plotting module included with several standard visualization functions

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(10, 5), dpi=80)

beam.parameters["pointing"] = np.array([0, 0, 1], dtype=np.float64)
beam.parameters["frequency"] = 930e6
beam.parameters["radius"] = 23.0

for ind, ax in enumerate(axes.flatten()):
    beam.parameters["frequency"] -= 100e6
    pyant.plotting.gain_heatmap(
        beam,
        resolution=301,
        min_elevation=80.0,
        ax=ax,
    )
    ax.set_title(f"frequency={beam.frequency/1e6:.1f} MHz")
fig.tight_layout()
```
