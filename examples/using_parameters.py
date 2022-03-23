'''
Using parameters
=================
'''
import numpy as np
import pyant

beam = pyant.Airy(
    azimuth=[0, 45.0, 0],
    elevation=[89.0, 80.0, 60.0],
    frequency=[930e6, 230e6],
    I0=10**4.81,
    radius=np.linspace(10, 23.0, num=20),
)

k = np.array([0, 0, 1.0]).T

# This is the shape and names of the parameters
print(f'beam.shape = {beam.shape}')
print(f'beam.parameters = {beam.parameters}')

# this means their values can be found trough the corresponding attributes
print(f'beam.radius = {beam.radius}')

# One needs to choose values for all parameters

# Either trough direct input into beam.gain
print(f'G = {beam.gain(k, pointing=k, frequency=314e6, radius=20.0)} ')
# pointing is the only parameter that supports also input by 
# azimuth and elevation
G = beam.gain(k, azimuth=20.2, elevation=89.1, frequency=314e6, radius=20.0)
print(f'G = {G} ')


# Or trough indexing of the currently entered parameters
print(f'G = {beam.gain(k, ind=(0,1,10))} ')
# (indexing can also be done as a dict for more readability)
print(f'G = {beam.gain(k, ind=dict(pointing=0,frequency=1,radius=10))} ')


# Or a combination of both
print(f'G = {beam.gain(k, ind=(0,None,10), frequency=333e6)} ')


print('-- exceptions --')
# Inconsistencies raise value and type errors
# like supplying both a index and a value
try:
    print(f'G = {beam.gain(k, ind=(0,1,10), frequency=333e6)} ')
except Exception as e:
    print(f'Exception: "{e}"')

# or not giving values for parameters at all
try:
    print(f'G = {beam.gain(k)} ')
except Exception as e:
    print(f'Exception: "{e}"')

# or not giving enough values
try:
    print(f'G = {beam.gain(k, ind=(0,1))} ')
except Exception as e:
    print(f'Exception: "{e}"')


# or trying to index scalar parameters
beam.frequency = 930e6

# now the size will be None for this parameter
print(f'beam.shape = {beam.shape}')

# so indexing it will raise an error
try:
    print(f'G = {beam.gain(k, ind=(0,1,10))} ')
except Exception as e:
    print(f'Exception: "{e}"')

print('-- exceptions end --')

# while setting it to None will just use the parameter value
print(f'G = {beam.gain(k, ind=(0,None,10))} ')

# if you have all scalar parameters, no index needs to be supplied
beam.radius = 23
beam.pointing = k.copy()

print(f'G = {beam.gain(k)} ')

# this also works with size=1 parameters
beam.radius = [23]

print(f'G = {beam.gain(k)} ')
