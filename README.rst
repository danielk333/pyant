PyAnt
=====

PyAnt is a standardization for implementing radar gain pattern in Python. Nothing more, nothing less.


Getting started
-----------------

To install (Not yet available)

.. code-block:: bash

   pip install pyant

or 

.. code-block:: bash

   git clone git@github.com:danielk333/pyant.git
   cd pyant
   pip install .

Then get started with then documentation at 


Example
---------

Plot the gain pattern of a dish radar modeled using the Cassegrain model.

.. code-block:: python

    import pyant

    ant = pyant.Cassegrain(
        azimuth=0,
        elevation=90.0, 
        frequency=930e6,
        I0=10**4.81,
        a0=23.0,
        a1=40.0,
    )

    pyant.plotting.gain_heatmap(ant, resolution=301, min_elevation=80.0)
    pyant.plotting.show()