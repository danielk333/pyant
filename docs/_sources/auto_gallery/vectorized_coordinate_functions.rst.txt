.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        Click :ref:`here <sphx_glr_download_auto_gallery_vectorized_coordinate_functions.py>`     to download the full example code
    .. rst-class:: sphx-glr-example-title

    .. _sphx_glr_auto_gallery_vectorized_coordinate_functions.py:


Vectorized coordinate functions
================================




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[0 1]
     [1 1]
     [0 0]]
    90.0
    45.00000000000001
    [90. 45.]
    [ 63.17540041  60.09108103 116.85216292 127.09584399  78.397541
      57.08846897 130.31341867  46.78115895 122.50599372 122.4710964
      70.81584609  99.47616363  62.1668323  114.23273139  85.66494047
      59.25677434  75.28794825  73.32477565 176.76361119  49.97238227
      63.19497384  69.22435712  89.50755228  62.99995827 147.95195633
      74.34316634  90.52317412 138.64714482  59.65924131 128.18434772
      67.80332322 100.73721247  41.72140911 143.29212276  76.27076978
      67.52419462  69.29738125 124.70961468  53.86053277  70.97333804
      42.27805585 101.77661457  74.8816984   78.64661537  64.14287648
     125.87997697  60.89317221  34.45004273 143.66819253 131.32838142
      53.60580414 123.77564557 155.30188427  88.50624804 137.09124555
     135.17482062  76.83767872 135.46449728 105.09062804  36.5622093
      83.67941534  74.0723582   68.87777706 133.89725549 134.78655632
      83.69828127 134.4699616   53.8628076   72.17131529  68.66811458
      81.55826904 113.29776567  91.76836824  56.48154111  80.00182459
      95.73899719  98.27862432 165.05831078  64.39999861 129.18073423
      31.19843888 157.56630353  45.70558045 125.51962715  71.28847545
      85.78256767  61.47044178  79.98299227 137.10038053 121.75283195
      58.15146938  74.24278001 127.30286858  41.55575885  90.4351392
      86.38354058  81.02002842  59.9049865   97.4145924   31.87717261]
    "vector_angle" (100) loop       performance: 1.9e+00 seconds
    "vector_angle" (100) vectorized performance: 2.5e-02 seconds
    [[  0. 120.]
     [ 45.  85.]
     [  1.   1.]]
    [0.         0.70710678 0.70710678]
    [ 0.07547909 -0.04357787  0.9961947 ]
    [[ 0.          0.07547909]
     [ 0.70710678 -0.04357787]
     [ 0.70710678  0.9961947 ]]
    [[-1.23549360e-02  1.53292878e-03  1.80150415e-02 -5.17233224e-02
      -3.55007884e-02 -2.96968232e-02 -1.54579462e-02  4.17432277e-03
       2.80504191e-03  1.68162868e-02]
     [ 9.99842714e-01  9.99998305e-01  9.99827986e-01  9.98542983e-01
       9.98995065e-01  9.99361555e-01  9.99861286e-01  9.99990925e-01
       9.99944917e-01  9.99811880e-01]
     [ 1.27240791e-02  1.02008612e-03 -4.41104611e-03  1.53820976e-02
       2.73597253e-02  1.98640797e-02 -6.20166179e-03  8.51557023e-04
       1.01140611e-02  9.66527779e-03]]
    "sph_to_cart" (100) loop       performance: 1.3e+00 seconds
    "sph_to_cart" (100) vectorized performance: 1.4e-02 seconds
    [[1. 0.]
     [1. 0.]
     [1. 1.]]
    [45.         35.26438968  1.73205081]
    [ 0. 90.  1.]
    [[45.          0.        ]
     [35.26438968 90.        ]
     [ 1.73205081  1.        ]]
    [[ 12.00147631 127.13597827 -54.72624762 224.86068937 -28.91443764
       40.6179913   95.20129549 -58.32434579 -57.19350229  -9.23281406]
     [ -1.30339108 -12.21245847  46.81163945 -11.83591143  15.45723279
       79.6774592  -54.1334088   11.52834004  32.86522289 -67.3340315 ]
     [  0.95025735   1.16446234   1.05763291   3.10849213   1.446238
        1.38574186   2.23266086   1.27806676   3.26410424   1.42409469]]
    "cart_to_sph" (100) loop       performance: 1.2e+00 seconds
    "cart_to_sph" (100) vectorized performance: 3.5e-02 seconds






|


.. code-block:: default

    import numpy as np
    import pyant

    import timeit

    number = 1000
    size = 100

    def coordinates_vector_angle():
        a = np.array([1,0,0])
        b = np.array([0,1,0])
        c = np.array([1,1,0])
        bc = np.append(b.reshape(3,1),c.reshape(3,1),axis=1)
        print(bc)

        th_ab = pyant.coordinates.vector_angle(a,b)
        print(th_ab)

        th_ac = pyant.coordinates.vector_angle(a,c)
        print(th_ac)

        th_abc = pyant.coordinates.vector_angle(a,bc)
        print(th_abc)

        x = np.random.randn(3,100)

        th_ax = pyant.coordinates.vector_angle(a,x)
        print(th_ax)
        # As opposed to
        #
        # for y in x.T:
        #     print(pyant.coordinates.vector_angle(a,y))

        dt_l = timeit.timeit(
            '''
    for y in x.T:
        pyant.coordinates.vector_angle(a,y)
    ''', 
            setup=
            f'''
    import pyant
    import numpy as np
    a = np.array([1,0,0])
    x = np.random.randn(3,{size})
        ''', 
            number = number,
        )

        dt_v = timeit.timeit(
            'pyant.coordinates.vector_angle(a,x)', 
            setup=
        f'''
    import pyant
    import numpy as np
    a = np.array([1,0,0])
    x = np.random.randn(3,{size})
        ''', 
            number = number,
        )
        print(f'"vector_angle" ({size}) loop       performance: {dt_l:.1e} seconds')
        print(f'"vector_angle" ({size}) vectorized performance: {dt_v:.1e} seconds')



    def coordinates_sph_to_cart():
        a = np.array([0,45,1], dtype=np.float64)
        b = np.array([120,85,1], dtype=np.float64)
        ab = np.append(a.reshape(3,1),b.reshape(3,1),axis=1)
        print(ab)

        sph_a = pyant.coordinates.sph_to_cart(a)
        print(sph_a)

        sph_b = pyant.coordinates.sph_to_cart(b)
        print(sph_b)

        sph_ab = pyant.coordinates.sph_to_cart(ab)
        print(sph_ab)

        x = np.random.randn(3,10)
        x[2,:] = 1.0

        sph_x = pyant.coordinates.sph_to_cart(x)
        print(sph_x)
        # As opposed to
        #
        # for y in x.T:
        #     print(pyant.coordinates.sph_to_cart(y))

        setup = f'''
    import pyant
    import numpy as np
    x = np.random.randn(3,{size})
    x[2,:] = 1.0
    '''

        dt_l = timeit.timeit(
        '''
    for y in x.T:
        pyant.coordinates.sph_to_cart(y)
        ''', 
            setup = setup, 
            number = number,
        )

        dt_v = timeit.timeit(
            'pyant.coordinates.sph_to_cart(x)', 
            setup = setup, 
            number = number,
        )
        print(f'"sph_to_cart" ({size}) loop       performance: {dt_l:.1e} seconds')
        print(f'"sph_to_cart" ({size}) vectorized performance: {dt_v:.1e} seconds')



    def coordinates_cart_to_sph():
        a = np.array([1,1,1], dtype=np.float64)
        b = np.array([0,0,1], dtype=np.float64)
        ab = np.append(a.reshape(3,1),b.reshape(3,1),axis=1)
        print(ab)

        sph_a = pyant.coordinates.cart_to_sph(a)
        print(sph_a)

        sph_b = pyant.coordinates.cart_to_sph(b)
        print(sph_b)

        sph_ab = pyant.coordinates.cart_to_sph(ab)
        print(sph_ab)

        x = np.random.randn(3,10)

        sph_x = pyant.coordinates.cart_to_sph(x)
        print(sph_x)
        # As opposed to
        #
        # for y in x.T:
        #     print(pyant.coordinates.cart_to_sph(y))

        setup = f'''
    import pyant
    import numpy as np
    x = np.random.randn(3,{size})
    '''

        dt_l = timeit.timeit(
        '''
    for y in x.T:
        pyant.coordinates.cart_to_sph(y)
        ''', 
            setup = setup, 
            number = number,
        )

        dt_v = timeit.timeit(
            'pyant.coordinates.cart_to_sph(x)', 
            setup = setup, 
            number = number,
        )
        print(f'"cart_to_sph" ({size}) loop       performance: {dt_l:.1e} seconds')
        print(f'"cart_to_sph" ({size}) vectorized performance: {dt_v:.1e} seconds')



    coordinates_vector_angle()
    coordinates_sph_to_cart()
    coordinates_cart_to_sph()

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  4.527 seconds)


.. _sphx_glr_download_auto_gallery_vectorized_coordinate_functions.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download sphx-glr-download-python

     :download:`Download Python source code: vectorized_coordinate_functions.py <vectorized_coordinate_functions.py>`



  .. container:: sphx-glr-download sphx-glr-download-jupyter

     :download:`Download Jupyter notebook: vectorized_coordinate_functions.ipynb <vectorized_coordinate_functions.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
