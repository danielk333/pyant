'''
Vectorized plotting
========================
'''
import time
import pyant

ant = pyant.Airy(
    azimuth=45,
    elevation=75.0, 
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
)


start_time = time.time()
pyant.plotting.gain_heatmap(ant, resolution=300, min_elevation=80.0, vectorized=False)
print(f'"gain_heatmap" ({300**2}) loop       performance: {time.time() - start_time:.1e} seconds')

start_time = time.time()
pyant.plotting.gain_heatmap(ant, resolution=300, min_elevation=80.0, vectorized=True)
print(f'"gain_heatmap" ({300**2}) vectorized performance: {time.time() - start_time:.1e} seconds')


pyant.plotting.show()