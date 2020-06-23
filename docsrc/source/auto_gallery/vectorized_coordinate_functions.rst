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
    [ 17.9803033  108.41985708  61.48280465  62.09899472  20.97938692
     158.42508375 117.01733794 129.00483553  33.40925949  49.04873367
     137.91449595 102.05077726  80.4491431  146.83933068 118.22063016
      61.33754322  63.73132131 127.01156106   9.11192459 109.19168275
      76.24190804  42.44934463 152.52203704 140.3932413   95.63844023
      49.20049082  39.41322288 135.08663867  75.59060564 114.45840203
     151.88383949  17.84448359 108.58595241  76.13133174  39.16665191
      51.16726527  89.31298951 138.29510317 114.78104615  99.55375715
     170.66831943 102.64233294 105.27162627  81.78722697  88.39090514
     142.50103215  52.86332897 129.20093302 120.78588025 155.02371363
     114.3359583  149.04127046 157.37049853 113.76210212  67.43502584
      44.99050879  97.79673725  24.50697303  70.43975197  28.38469838
      29.1940539  132.47270419  78.41486657  65.72870628 100.58686299
     164.20833132  46.47628384  97.72302976  52.12996719  83.3264601
     110.68701219  84.08459147  53.46678074 100.94091148 108.23924668
     110.37279709  22.23451298  52.75516627 158.95272496 140.05868016
      65.79346887  87.99474456 116.24680379 121.38700785 126.14327068
      71.44438721  59.56891491  80.48423907  49.52944894  52.91839309
      95.00243142  88.59590912  75.40877836  50.41599195  27.03598093
     107.76312042 135.7167363   55.81082806  43.46267817  37.27794816]
    "vector_angle" (100) loop       performance: 2.0e+00 seconds
    "vector_angle" (100) vectorized performance: 2.5e-02 seconds
    [[  0. 120.]
     [ 45.  85.]
     [  1.   1.]]
    [0.         0.70710678 0.70710678]
    [ 0.07547909 -0.04357787  0.9961947 ]
    [[ 0.          0.07547909]
     [ 0.70710678 -0.04357787]
     [ 0.70710678  0.9961947 ]]
    [[ 0.02044639  0.02509242  0.00604926 -0.02886215 -0.01966093  0.00230625
      -0.00813496  0.00401154 -0.01401502  0.00472753]
     [ 0.99953081  0.99938604  0.99997943  0.99937367  0.99972997  0.99945888
       0.99989276  0.99946424  0.9999007   0.99998734]
     [-0.02280583 -0.02445232 -0.00213289  0.02047542  0.01238652 -0.03281213
      -0.01217746 -0.03248278 -0.0014741  -0.00172185]]
    "sph_to_cart" (100) loop       performance: 1.3e+00 seconds
    "sph_to_cart" (100) vectorized performance: 1.5e-02 seconds
    [[1. 0.]
     [1. 0.]
     [1. 1.]]
    [45.         35.26438968  1.73205081]
    [ 0. 90.  1.]
    [[45.          0.        ]
     [35.26438968 90.        ]
     [ 1.73205081  1.        ]]
    [[ 54.56087546  -3.62161631 217.9653295  122.61560424  92.32910271
      -66.81652366  93.20493556 -64.72454439 -69.36568579 191.74351267]
     [ 10.27440341  28.30549616  -3.46127214  50.79621929  35.87332779
       12.47564264 -12.34977886 -62.13926921  16.30227739  69.36576588]
     [  1.2103633    1.1489424    1.13526047   1.64665088   0.74140168
        3.29496016   1.86100629   1.84363529   2.74295241   2.02396473]]
    "cart_to_sph" (100) loop       performance: 1.3e+00 seconds
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

   **Total running time of the script:** ( 0 minutes  4.640 seconds)


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
