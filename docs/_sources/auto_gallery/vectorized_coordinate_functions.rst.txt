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
    [ 68.06060928  64.44403004 147.22335708  40.86841157 103.03787612
      99.96185124  56.36159355 132.80361456  51.31069176  62.9607813
     140.63792965  92.48069822 146.22454306 100.46561485 119.12442932
      79.58122085 143.69342     93.98432507  47.10018076 149.01603946
      84.2026655   84.15221942 110.76212206 131.51805764  72.84768235
      72.15243454  84.53764055 132.7545784  113.22081922 155.45535
     109.9655733   16.31425561 109.10238879 155.02710471  89.48765473
     104.474103    66.96965137  79.35592525 162.39050327  23.52699885
      67.33590576 129.94222564 142.45500497  39.17167048 104.39235605
      70.09864267  32.00380595  21.79303941  91.67959653  92.20740787
      46.55610414  94.32447688 130.08399461  76.73324824  73.6694332
     104.49871304  28.276935    49.44338738  11.33264329  19.62253055
     102.58751784  48.17204362 122.25170423  27.71685467 111.05165311
     138.52157002 105.93881388  76.0273971  138.98296019  49.0161775
      57.01961445  84.34700506  90.3920307  138.06219816  56.70562863
     125.28685401  51.16011031 174.80120075  93.67771216  16.45021705
      93.84734492 140.35837706 121.75423173  19.88484971 113.26008097
     135.249041    89.55794255  41.6750644  170.37425903  62.80913201
      32.0109425   49.31021391  80.02503748  74.70109777 170.40158538
      92.29135597 148.8915862  144.74054351  64.005617    91.23803867]
    "vector_angle" (100) loop       performance: 2.0e+00 seconds
    "vector_angle" (100) vectorized performance: 2.6e-02 seconds
    [[  0. 120.]
     [ 45.  85.]
     [  1.   1.]]
    [0.         0.70710678 0.70710678]
    [ 0.07547909 -0.04357787  0.9961947 ]
    [[ 0.          0.07547909]
     [ 0.70710678 -0.04357787]
     [ 0.70710678  0.9961947 ]]
    [[ 0.01873133 -0.00432245  0.01405931  0.01279095 -0.02387233  0.00640837
       0.00374267 -0.01943662 -0.00201378  0.01439904]
     [ 0.99959601  0.99998879  0.99970646  0.99990262  0.9997109   0.99989588
       0.99994815  0.99968299  0.9998922   0.99959852]
     [-0.02137633  0.00193179 -0.01973122 -0.00558051 -0.00286714  0.01292879
       0.00947014  0.01600415  0.01454418 -0.02440224]]
    "sph_to_cart" (100) loop       performance: 1.3e+00 seconds
    "sph_to_cart" (100) vectorized performance: 1.7e-02 seconds
    [[1. 0.]
     [1. 0.]
     [1. 1.]]
    [45.         35.26438968  1.73205081]
    [ 0. 90.  1.]
    [[45.          0.        ]
     [35.26438968 90.        ]
     [ 1.73205081  1.        ]]
    [[226.14377802 221.30259875 241.05375502  40.6411154  105.00483419
       49.53843752 101.77811961 -77.36919229 -47.24219298  90.75065679]
     [-29.35815468  69.78580258   9.7067689    2.13984389  62.70191109
       25.99131943  21.16078065 -22.16943226 -32.60091889  58.51654524]
     [  0.41821126   2.05432133   2.49527324   1.447207     0.98560243
        2.1958699    1.32342451   2.63511118   0.2557918    0.42271237]]
    "cart_to_sph" (100) loop       performance: 1.3e+00 seconds
    "cart_to_sph" (100) vectorized performance: 4.0e-02 seconds






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

   **Total running time of the script:** ( 0 minutes  4.694 seconds)


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
