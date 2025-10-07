# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# # Vectorized coordinate functions

import timeit


number = 100
size = 1000


def coordinates_vector_angle():
    dt_l = timeit.timeit(
        """
for y in x.T:
    pyant.coordinates.vector_angle(a,y)
""",
        setup=f"""
import pyant
import numpy as np
a = np.array([1,0,0])
x = np.random.randn(3,{size})
    """,
        number=number,
    )

    dt_v = timeit.timeit(
        "pyant.coordinates.vector_angle(a,x)",
        setup=f"""
import pyant
import numpy as np
a = np.array([1,0,0])
x = np.random.randn(3,{size})
    """,
        number=number,
    )
    print(f'"vector_angle"({size}) loop       performance: {dt_l:.1e} seconds')
    print(f'"vector_angle"({size}) vectorized performance: {dt_v:.1e} seconds')
    print(f"vectorized speedup = {dt_l/dt_v}")


def coordinates_sph_to_cart():
    setup = f"""
import pyant
import numpy as np
x = np.random.randn(3,{size})
x[2,:] = 1.0
"""

    dt_l = timeit.timeit(
        """
for y in x.T:
    pyant.coordinates.sph_to_cart(y)
    """,
        setup=setup,
        number=number,
    )

    dt_v = timeit.timeit(
        "pyant.coordinates.sph_to_cart(x)",
        setup=setup,
        number=number,
    )
    print(f'"sph_to_cart" ({size}) loop       performance: {dt_l:.1e} seconds')
    print(f'"sph_to_cart" ({size}) vectorized performance: {dt_v:.1e} seconds')
    print(f"vectorized speedup = {dt_l/dt_v}")


def coordinates_cart_to_sph():
    setup = f"""
import pyant
import numpy as np
x = np.random.randn(3,{size})
"""

    dt_l = timeit.timeit(
        """
for y in x.T:
    pyant.coordinates.cart_to_sph(y)
    """,
        setup=setup,
        number=number,
    )

    dt_v = timeit.timeit(
        "pyant.coordinates.cart_to_sph(x)",
        setup=setup,
        number=number,
    )
    print(f'"cart_to_sph" ({size}) loop       performance: {dt_l:.1e} seconds')
    print(f'"cart_to_sph" ({size}) vectorized performance: {dt_v:.1e} seconds')
    print(f"vectorized speedup = {dt_l/dt_v}")


coordinates_vector_angle()
coordinates_sph_to_cart()
coordinates_cart_to_sph()
