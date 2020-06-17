'''
Omnidirectional antenna
========================
'''

import pyant

import numpy as np

class Omni(pyant.Beam):
    def gain(self,k):
        return 1.0

ant = Omni(
    azimuth=0.0, 
    elevation=90.0, 
    frequency=47e6,
)

print(ant.gain([0,0,1]))