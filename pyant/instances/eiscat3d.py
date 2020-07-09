#!/usr/bin/env python

'''A collection of functions and information for the EISCAT 3D Radar system.

'''

#Python standard import
try:
    import importlib.resources as ilibr
except ImportError:
    ilibr = None

import numpy as np
import scipy.constants

e3d_frequency = 233e6
e3d_antenna_gain = 10.0**0.3 #3 dB peak antenna gain?


def e3d_subarray(freqeuncy):
    '''Generate cartesian positions `x,y,z` in meters of antenna elements in one standard EISCAT 3D subarray.

    #TODO: Sphinx params doc
    '''
    l0 = scipy.constants.c/freqeuncy;

    dx = 1.0/np.sqrt(3);
    dy = 0.5;

    xall = []
    yall = []

    x0 = np.array([np.arange(-2.5,-5.5,-.5).tolist() + np.arange(-4.5,-2.0,.5).tolist()])[0]*dx
    y0 = np.arange(-5,6,1)*dy

    for iy in range(11):
        nx = 11-np.abs(iy-5)
        x_now = x0[iy]+np.array(range(nx))*dx
        y_now = y0[iy]+np.array([0.0]*(nx))
        xall += x_now.tolist()
        yall += y_now.tolist()

    x = l0*np.array(xall);
    y = l0*np.array(yall);
    z = x*0.0;

    return x,y,z


def e3d_array(freqeuncy, fname=None, configuration='full'):
    '''Generate the antenna positions for a EISCAT 3D Site based on submodule positions of a file.

    #TODO: Sphinx params doc
    '''
    
    def _read_e3d_submodule_pos(path):
        dat = []
        with open(path,'r') as file:
            for line in file:
                dat.append( list(map(lambda x: float(x),line.split() )) )
        dat = np.array(dat)
        return dat


    if fname is None:
        with ilibr.path('pyant.instances.data', 'e3d_subgroup_positions.txt') as pth:
            dat = _read_e3d_submodule_pos(pth)
    else:
        dat = _read_e3d_submodule_pos(fname)

    sx,sy,sz = e3d_subarray(freqeuncy)

    if configuration == 'full':
        pass
    elif configuration == 'half-dense':
        dat = dat[ ( np.sum(dat**2.0,axis=1) < 27.0**2.0 ) ,: ]
    elif configuration == 'half-sparse':
        dat = dat[ \
        np.logical_or( \
            np.logical_or(\
                np.logical_and( np.sum(dat**2,axis=1) < 10**2 , np.sum(dat**2,axis=1) > 7**2 ), \
                np.logical_and( np.sum(dat**2,axis=1) < 22**2 , np.sum(dat**2,axis=1) > 17**2 )),  \
            np.logical_and( np.sum(dat**2,axis=1) < 36**2 , np.sum(dat**2,axis=1) > 30**2 ) \
        ),: ]
    elif configuration == 'module':
        dat = np.zeros((1,2))

    antennas = np.zeros((3, len(sx), dat.shape[0]), dtype=dat.dtype)
    for i in range(dat.shape[0]):
        for j in range(len(sx)):
            antennas[0,j,i] = sx[j] + dat[i,0]
            antennas[1,j,i] = sy[j] + dat[i,1]
            antennas[2,j,i] = sz[j]
    return antennas

