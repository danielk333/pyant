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


import numpy as np
import pyant
import timeit


number = 1000
size = 100


def coordinates_vector_angle():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = np.array([1, 1, 0])
    bc = np.append(b.reshape(3, 1), c.reshape(3, 1), axis=1)
    print(bc)

    th_ab = pyant.coordinates.vector_angle(a, b)
    print(th_ab)

    th_ac = pyant.coordinates.vector_angle(a, c)
    print(th_ac)

    th_abc = pyant.coordinates.vector_angle(a, bc)
    print(th_abc)

    x = np.random.randn(3, 100)

    th_ax = pyant.coordinates.vector_angle(a, x)
    print(th_ax)
    # As opposed to
    #
    # for y in x.T:
    #     print(pyant.coordinates.vector_angle(a,y))

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
    a = np.array([0, 45, 1], dtype=np.float64)
    b = np.array([120, 85, 1], dtype=np.float64)
    ab = np.append(a.reshape(3, 1), b.reshape(3, 1), axis=1)
    print(ab)

    sph_a = pyant.coordinates.sph_to_cart(a)
    print(sph_a)

    sph_b = pyant.coordinates.sph_to_cart(b)
    print(sph_b)

    sph_ab = pyant.coordinates.sph_to_cart(ab)
    print(sph_ab)

    x = np.random.randn(3, 10)
    x[2, :] = 1.0

    sph_x = pyant.coordinates.sph_to_cart(x)
    print(sph_x)
    # As opposed to
    #
    # for y in x.T:
    #     print(pyant.coordinates.sph_to_cart(y))

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
    a = np.array([1, 1, 1], dtype=np.float64)
    b = np.array([0, 0, 1], dtype=np.float64)
    ab = np.append(a.reshape(3, 1), b.reshape(3, 1), axis=1)
    print(ab)

    sph_a = pyant.coordinates.cart_to_sph(a)
    print(sph_a)

    sph_b = pyant.coordinates.cart_to_sph(b)
    print(sph_b)

    sph_ab = pyant.coordinates.cart_to_sph(ab)
    print(sph_ab)

    x = np.random.randn(3, 10)

    sph_x = pyant.coordinates.cart_to_sph(x)
    print(sph_x)
    # As opposed to
    #
    # for y in x.T:
    #     print(pyant.coordinates.cart_to_sph(y))

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
