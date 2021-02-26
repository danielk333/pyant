'''
Vectorized beams
========================
'''
import time
import pyant

import numpy as np

num = 10000

ks = np.zeros((3, num), dtype=np.float64)
ks[0,:] = np.linspace(-0.5, 0.5, num=num)
ks[2,:] = np.sqrt(1 - ks[0,:]**2 - ks[1,:]**2)
fs = np.linspace(200e6, 930e6, num=num)


start_time = time.time()

ant = pyant.Airy(
    azimuth=45,
    elevation=75.0, 
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
)
g = np.empty((num,), dtype=np.float64)
for ind in range(num):
    ant.frequency = fs[ind]
    g[ind] = ant.gain(ks[:, ind])

print(f'g.shape={g.shape}')
execution_time_loop = time.time() - start_time
print(f'"gain calculations" ({num}) loop       performance: {execution_time_loop:.1e} seconds')

start_time = time.time()

g = ant.gain(
    ks,
    frequency=fs,
    vectorized_parameters=True,
)
print(f'g.shape={g.shape}')
execution_time_vectorized = time.time() - start_time
print(f'"gain calculations" ({num}) vectorized performance: {execution_time_vectorized:.1e} seconds')

#This way also works
start_time = time.time()

ant.frequency = fs
g = ant.gain(
    ks,
    vectorized_parameters=True,
)
print(f'g.shape={g.shape}')
print(f'"gain calculations" ({num}) vectorized performance: {time.time() - start_time:.1e} seconds')


print(f'Speedup = {execution_time_loop/execution_time_vectorized:.2f}')

pyant.plotting.show()