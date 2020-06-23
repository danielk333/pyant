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
    [152.47520052  47.65824928  15.77639954  70.6832322   47.73320281
      44.57121559  76.04631964 146.2230476   19.32157762  35.53330669
      82.58057756  46.79562144  45.26308388 127.75966112  28.34830136
      85.13451933  82.70007112  76.41429248  94.21526109  46.81321588
      63.4198988  100.37858952  64.23506443  52.00542242  30.38817335
     111.8005347  102.4585908   63.75332836  95.81801057 111.04522015
     118.63948007 128.29211131  98.81628029  80.19663372  92.63441245
      47.0582532   55.69015812 102.10403371 171.77749475  53.80562846
      50.06677944  83.64411297  77.52259282  70.96124981 163.18719176
      56.51867033 106.93622945  27.91679013 131.42675596  49.29558425
      51.9865435   40.47066083 127.96462255 116.51786838  63.84334247
      82.56581016  75.76065906  71.77409521  29.70319225 118.18346088
     116.807051    58.75315923 145.81898391  35.92113462 155.32593807
     108.18463045  49.35887079  76.94905544  45.43613689  72.13987538
      60.65111018 163.67691805 160.81749545  97.50217545 119.49741961
     145.94232499 146.46117313  65.21301073  63.75946921 114.57320645
      69.01479682  61.33181562  39.34163114 139.31930522 118.86888362
      53.63885955  72.85461379  75.91575223  52.81043578  45.31713763
      71.96422111  57.0232131  167.62351902  92.73360821 111.10078451
     138.15617458  78.03403213  19.25598619  41.95633721  97.88513614]
    "vector_angle" (100) loop       performance: 1.9e+00 seconds
    "vector_angle" (100) vectorized performance: 3.0e-02 seconds
    [[  0. 120.]
     [ 45.  85.]
     [  1.   1.]]
    [0.         0.70710678 0.70710678]
    [ 0.07547909 -0.04357787  0.9961947 ]
    [[ 0.          0.07547909]
     [ 0.70710678 -0.04357787]
     [ 0.70710678  0.9961947 ]]
    [[-0.0118186   0.01130735  0.0125578  -0.03632938 -0.01837481 -0.00676333
      -0.01555383 -0.01421832 -0.02492175  0.02725418]
     [ 0.99989547  0.99988428  0.99991021  0.99913855  0.99982529  0.99997471
       0.99945922  0.99985928  0.9996888   0.99892594]
     [-0.00832923 -0.01017735  0.00467729 -0.02005839  0.00342867  0.00220043
       0.02897143 -0.00890324  0.00109943 -0.03747232]]
    "sph_to_cart" (100) loop       performance: 1.4e+00 seconds
    "sph_to_cart" (100) vectorized performance: 1.5e-02 seconds
    [[1. 0.]
     [1. 0.]
     [1. 1.]]
    [45.         35.26438968  1.73205081]
    [ 0. 90.  1.]
    [[45.          0.        ]
     [35.26438968 90.        ]
     [ 1.73205081  1.        ]]
    [[260.26489168 204.92159192 138.95268341  -8.19873344  76.68906573
       85.68191755 -79.46666999 175.35936443  64.74577274  74.51318934]
     [ -7.32443814  36.17585313 -32.31822633  56.7131808   31.5251008
       18.31309289  17.58387806 -19.29388185 -21.01528605 -19.08003683]
     [  2.06417287   1.81673157   2.09217726   0.49446199   1.91455158
        2.85174772   1.13305975   2.00473059   0.67192628   1.82019826]]
    "cart_to_sph" (100) loop       performance: 1.3e+00 seconds
    "cart_to_sph" (100) vectorized performance: 3.6e-02 seconds






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

   **Total running time of the script:** ( 0 minutes  4.710 seconds)


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
