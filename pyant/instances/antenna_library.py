#!/usr/bin/env python

'''A collection of functions that return common instances of the :class:`~antenna.BeamPattern` class.

Contains for example:
 * Uniformly filled circular aperture of radius a
 * Cassegrain antenna with radius a0 and subreflector radius a1
 * Planar gaussian illuminated aperture (approximates a phased array)

Reference:
https://www.cv.nrao.edu/course/astr534/2DApertures.html
'''
import os
#import pdb

import numpy as np
import scipy.constants
import scipy.special
import scipy.interpolate
import h5py





def uhf_meas(k_in,beam):
    '''Measured UHF beam pattern

    '''
    theta = coord.angle_deg(beam.on_axis,k_in)
    # scale beam width by frequency
    sf=beam.f/930e6
    
    return(beam.I_0*beam.gf(sf*np.abs(theta)))








def airy_beam(az0, el0, I_0, f, a):
    '''# TODO: Description.

    '''
    beam = antenna.BeamPattern(airy, az0, el0, I_0, f, beam_name='Airy')
    beam.a = a
    return beam


def uhf_beam(az0, el0, I_0, f, beam_name='UHF Measured beam'):
    '''# TODO: Description.

    '''
    beam = antenna.BeamPattern(uhf_meas, az0, el0, I_0, f, beam_name=beam_name)

    bmod=np.genfromtxt("data/bp.txt")
    angle=bmod[:,0]
    gain=10**(bmod[:,1]/10.0)
    gf=sio.interp1d(np.abs(angle),gain)
    
    beam.gf = gf
    return beam


def cassegrain_beam(az0, el0, I_0, f, a0, a1, beam_name="Cassegrain"):
    '''# TODO: Description.

    az and el of on-axis
    lat and lon of location
    I_0 gain on-axis
    a0 diameter of main reflector
    a1 diameter of the subreflector
    '''
    beam = antenna.BeamPattern(cassegrain, az0, el0, I_0, f, beam_name=beam_name)
    beam.a0 = a0
    beam.a1 = a1
    return beam

def planar_beam(az0, el0, I_0, f, a0, az1, el1):
    '''# TODO: Description.

    '''
    beam = antenna.BeamPattern(planar, az0, el0, I_0, f, beam_name='Planar')
    beam.a0 = a0
    beam.plane_normal=coord.azel_to_cart(az1, el1, 1.0)
    beam.lam=c.c/f
    beam.point(az0,el0)
    return beam






def tsr_fence_beam(f = 224.0e6):
    a = 30               # Panel width, metres (30 = 1 panel, 120 = all panels)
    b = 40               # Panel height, metres
    c = 299792458        # Speed of light, m/s
    wavelength = c/f     # Wavelength, metres

    ar = a / wavelength  # Antenna size in wavelengths
    br = b / wavelength  # ditto

    # Make an equirectangular projection mesh (2000 points per axis)
    x = np.linspace(-np.pi/2,np.pi/2,4000)
    y = np.linspace(-np.pi/2,np.pi/2,4000)
    xx,yy = np.meshgrid(x,y)

    # Calclate the beam pattern
    z = unidirectional_broadside_rectangular_array(ar,br,xx,yy)

    # Normalise (4pi steradian * num.pixels / integrated gain / pi^2)
    scale = 4 * np.pi * z.size / np.sum(z)   # Normalise over sphere
    sincint = np.pi*np.pi                    # Integral of the sinc^2()s: -inf:inf

    els = [30.0, 60.0, 90.0, 60.0]
    azs = [0.0, 0.0, 0.0, 180.0]

    def TSR_fence_gain(k_in, beam):
        G = 0.0

        for az, el in zip(azs, els):
            G += TSR_gain_point(k_in, beam, az + beam.az0, el + beam.el0 - 90.0)

        return G

    beam = antenna.BeamPattern(TSR_fence_gain, az0=0.0, el0=90.0, I_0=scale/sincint, f=f, beam_name='Tromso Space Radar Fence Beam')
    beam.ar = ar
    beam.br = br

    return beam


def tsr_beam(el0, f = 224.0e6):
    a = 120              # Panel width, metres (30 = 1 panel, 120 = all panels)
    b = 40               # Panel height, metres
    c = 299792458        # Speed of light, m/s
    wavelength = c/f     # Wavelength, metres

    ar = a / wavelength  # Antenna size in wavelengths
    br = b / wavelength  # ditto

    # Make an equirectangular projection mesh (2000 points per axis)
    x = np.linspace(-np.pi/2,np.pi/2,4000)
    y = np.linspace(-np.pi/2,np.pi/2,4000)
    xx,yy = np.meshgrid(x,y)

    # Calclate the beam pattern
    z = unidirectional_broadside_rectangular_array(ar,br,xx,yy)

    # Normalise (4pi steradian * num.pixels / integrated gain / pi^2)
    scale = 4 * np.pi * z.size / np.sum(z)   # Normalise over sphere
    sincint = np.pi*np.pi                    # Integral of the sinc^2()s: -inf:inf

    beam = antenna.BeamPattern(TSR_gain, az0=0.0, el0=el0, I_0=scale/sincint, f=f, beam_name='Tromso Space Radar Beam')
    beam.ar = ar
    beam.br = br

    return beam