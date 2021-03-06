'''
Vectorized gain functions
================================
'''
import time

import numpy as np
import pyant


number = 1000
kn = 1000

ant = pyant.Airy(
    azimuth=45.0,
    elevation=75.0, 
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
)

kx = np.linspace(-1, 1, num=kn)
ky = np.linspace(-1, 1, num=kn)

size = len(kx)*len(ky)

#loop version
start_time = time.time()

G = np.zeros((len(kx),len(ky)))
for i,x in enumerate(kx):
    for j,y in enumerate(ky):
        k=np.array([x, y, np.sqrt(1.0 - x**2 + y**2)])
        G[i,j] = ant.gain(k)

loop_time = time.time() - start_time

#vectorized version
start_time = time.time()

xv, yv = np.meshgrid(kx, ky, sparse=False, indexing='ij')
k = np.empty((3,size), dtype=np.float64)
k[0,:] = xv.reshape(1,size)
k[1,:] = yv.reshape(1,size)
k[2,:] = np.sqrt(1.0 - k[0,:]**2 + k[1,:]**2)

#We want to use reshape as a inverse function so we make sure its the exact same dimensionality
G = np.zeros((1,size))
G[0,:] = ant.gain(k)
G = G.reshape(len(kx),len(ky))

vector_time = time.time() - start_time

print(G)

print(f'"Airy.gain" ({size}) loop       performance: {loop_time:.1e} seconds')
print(f'"Airy.gain" ({size}) vectorized performance: {vector_time:.1e} seconds')
