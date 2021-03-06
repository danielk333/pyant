'''
Antenna array gain
===========================
'''
import time
import numpy as np

import pyant

xv, yv = np.meshgrid(np.linspace(-50,50, num=22), np.linspace(-50,50, num=22))
antennas = np.zeros((22**2, 3))
antennas[:,0] = xv.flatten()
antennas[:,1] = yv.flatten()

ant = pyant.Array(
    azimuth=0,
    elevation=90.0, 
    frequency=46.5e6,
    antennas=antennas,
)

## Uncomment these to try the speed up for more complex gain calculations

# start_time = time.time()
# pyant.plotting.gain_heatmap(ant, resolution=100, min_elevation=80.0, vectorized=False)
# print(f'"gain_heatmap" ({100**2}) loop       performance: {time.time() - start_time:.1e} seconds')

# start_time = time.time()
# pyant.plotting.gain_heatmap(ant, resolution=100, min_elevation=80.0, vectorized=True)
# print(f'"gain_heatmap" ({100**2}) vectorized performance: {time.time() - start_time:.1e} seconds')


pyant.plotting.gain_heatmap(ant, resolution=100, min_elevation=80.0)
pyant.plotting.show()