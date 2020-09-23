#!/usr/bin/env python

'''Useful coordinate related functions.

(c) 2020 Daniel Kastinen

With contributions from Tom Grydeland
'''

from .beam import Beam

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import functools
import operator

def show():
    '''Shorthand for matplotlib :code:`show` function.'''
    plt.show()


def antenna_configuration(antennas, ax=None, color=None):
    '''Plot the 3d antenna positions
    '''
    if ax is None:
        fig = plt.figure(figsize=(15,7))
        ax = fh.add_subplot(111, projection='3d')
    else:
        fh = None

    if color is None:
        style_ = '.'
    else:
        style_ = '.' + color

    ax.plot(
        antennas[:,0,:].flatten(), 
        antennas[:,1,:].flatten(), 
        antennas[:,2,:].flatten(),  
        style_
    )
    ax.set_title('Antennas', fontsize=22)
    ax.set_xlabel('X-position [m]', fontsize=20)
    ax.set_ylabel('Y-position [m]', fontsize=20)
    ax.set_zlabel('Z-position [m]', fontsize=20)

    return fh, ax


def gains(beam, resolution=1000, min_elevation = 0.0, alpha = 0.5, usetex=False):
    '''Plot the gain of a list of beam patterns as a function of elevation at :math:`0^\circ` degrees azimuth.
    
    :param beam: Beam or list of beams.
    :param int resolution: Number of points to divide the set elevation range into.
    :param float min_elevation: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`.
    '''
    #set TeX interperter
    plt.rc('text', usetex=usetex)

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111)
    
    
    theta=n.linspace(min_elevation,90.0,num=resolution)
    
    S=n.zeros((resolution,len(beams)))
    for b,beam in enumerate(beams):
        for i,th in enumerate(theta):
            k=coord.azel_to_cart(0.0, th, 1.0)
            S[i,b]=beam.gain(k)
    for b in range(len(beams)):
        ax.plot(90-theta,n.log10(S[:,b])*10.0,label="Gain " + beams[b].beam_name, alpha=alpha)
    ax.legend()
    bottom, top = plt.ylim()
    plt.ylim((0,top))
    ax.set_xlabel('Zenith angle [deg]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    if usetex:
        ax.set_ylabel('Gain $G$ [dB]',fontsize=24)
    else:
        ax.set_ylabel('Gain [dB]',fontsize=24)
    ax.set_title('Gain patterns',fontsize=28)

    return fig, ax



def gain_surface(beam, resolution=200, min_elevation = 0.0, usetex=False):
    '''Creates a 3d plot of the beam-patters as a function of azimuth and elevation in terms of wave vector ground projection coordinates.
    
    :param BeamPattern beam: Beam pattern to plot.
    :param int res: Number of points to devide the wave vector x and y component range into, total number of caluclation points is the square of this number.
    :param float min_elevation: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`. This number defines the half the length of the square that the gain is calculated over, i.e. :math:`\cos(el_{min})`.
    '''
    #set TeX interperter
    plt.rc('text', usetex=usetex)

    
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(111, projection='3d')
    
    
    kx=n.linspace(-n.cos(min_elevation*n.pi/180.0),n.cos(min_elevation*n.pi/180.0),num=res)
    ky=n.linspace(-n.cos(min_elevation*n.pi/180.0),n.cos(min_elevation*n.pi/180.0),num=res)
    
    S=n.zeros((res,res))
    K=n.zeros((res,res,2))
    for i,x in enumerate(kx):
        for j,y in enumerate(ky):
            z2 = x**2 + y**2
            if z2 < n.cos(min_elevation*n.pi/180.0)**2:
                k=n.array([x, y, n.sqrt(1.0 - z2)])
                S[i,j]=beam.gain(k)
            else:
                S[i,j] = 0;
            K[i,j,0]=x
            K[i,j,1]=y
    SdB = n.log10(S)*10.0
    SdB[SdB < 0] = 0
    surf = ax.plot_surface(K[:,:,0],K[:,:,1],SdB,cmap=cm.plasma, linewidth=0, antialiased=False, vmin=0, vmax=n.max(SdB))
    #surf = ax.plot_surface(K[:,:,0],K[:,:,1],S.T,cmap=cm.plasma,linewidth=0)
    if usetex:
        ax.set_xlabel('$k_x$ [1]',fontsize=24)
        ax.set_ylabel('$k_y$ [1]',fontsize=24)
        ax.set_zlabel('Gain $G$ [dB]',fontsize=24)
    else:
        ax.set_xlabel('kx [1]',fontsize=24)
        ax.set_ylabel('ky [1]',fontsize=24)
        ax.set_zlabel('Gain [dB]',fontsize=24)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.show()






def gain_heatmap(beam, polarization=None, resolution=201, min_elevation=0.0, levels=20, ax=None, vectorized=True, ind=None, usetex=False):
    '''Creates a heatmap of the beam-patterns as a function of azimuth and elevation in terms of wave vector ground projection coordinates.
    
    :param Beam/Beams beam: Beam pattern to plot.
    :param numpy.ndarray polarization: The polarization in terms of a Jones vector of the incoming waves.
    :param int resolution: Number of points to divide the wave vector x and y components into, total number of calculation points is the square of this number.
    :param float min_elevation: Minimum elevation in degrees, elevation range is from this number to :math:`90^\circ`. This number defines the half the length of the square that the gain is calculated over, i.e. :math:`\cos(el_{min})`.
    :param int levels: Number of levels in the contour plot.
    :param bool vectorized: Use vectorized gain functionality to calculate gain-map.
    :return: matplotlib axis and figure handles
    '''

    #set TeX interperter
    plt.rc('text', usetex=usetex)

    if ax is None:
        fig = plt.figure() # figsize=(15,7))
        ax = fig.add_subplot(111)
    else:
        fig = None

    if isinstance(beam, Beam):
        ind_, shape_ = beam.default_ind(ind)
        if shape_['pointing'] is not None:
            pointing = beam.pointing[:,ind_['pointing']]
        else:       
            pointing = beam.pointing
    elif isinstance(beam, list):
        pointing = np.array([0,0,1])
    else:
        raise TypeError(f'Can only plot Beam or Beams, not "{type(beam)}"')


    # We will draw a k-space circle centered on `pointing` with a radius of cos(min_elevation)
    # TODO: Limit to norm(k) <= 1 (horizon) since below-horizon will be discarded anyway
    kx=np.linspace(
        pointing[0] - np.cos(min_elevation*np.pi/180.0),
        pointing[0] + np.cos(min_elevation*np.pi/180.0),
        num=resolution,
    )
    ky=np.linspace(
        pointing[1] - np.cos(min_elevation*np.pi/180.0),
        pointing[1] + np.cos(min_elevation*np.pi/180.0),
        num=resolution,
    )
    
    
    K=np.zeros((resolution,resolution,2))

    if vectorized:
        K[:,:,0], K[:,:,1] = np.meshgrid(kx, ky, sparse=False, indexing='ij')
        size = resolution**2
        k = np.empty((3,size), dtype=np.float64)
        k[0,:] = K[:,:,0].reshape(1,size)
        k[1,:] = K[:,:,1].reshape(1,size)

        z2 = k[0,:]**2 + k[1,:]**2
        z2_c = (pointing[0] - k[0,:])**2 + (pointing[1] - k[1,:])**2

        inds_ = np.logical_and(z2_c < np.cos(min_elevation*np.pi/180.0)**2, z2 <= 1.0)
        not_inds_ = np.logical_not(inds_)

        k[2,inds_] = np.sqrt(1.0 - z2[inds_])
        k[2,not_inds_] = 0
        S = np.ones((1,size))
        if isinstance(beam, Beam):
            S[0,inds_] = beam.gain(k[:,inds_], polarization=polarization, ind=ind)
        elif isinstance(beam, list):
            S[0,inds_] = functools.reduce(operator.add, [b.gain(k[:,inds_], polarization=polarization, ind=ind) for b in beam])
        else:
            raise TypeError(f'Can only plot Beam or list, not "{type(beam)}"')
        
        S = S.reshape(resolution,resolution)

    else:
        S = np.ones((resolution,resolution))
        for i,x in enumerate(kx):
            for j,y in enumerate(ky):
                z2_c = (pointing[0]-x)**2 + (pointing[1]-y)**2
                z2 = x**2 + y**2
                if z2_c < np.cos(min_elevation*np.pi/180.0)**2 and z2 <= 1.0:

                    k=np.array([x, y, np.sqrt(1.0 - z2)])
                    if isinstance(beam, Beam):
                        S[i,j] = beam.gain(k, polarization=polarization, ind=ind)
                    elif isinstance(beam, list):
                        S[i,j] = functools.reduce(operator.add, [b.gain(k, polarization=polarization, ind=ind) for b in beam])
                    else:
                        raise TypeError(f'Can only plot Beam or list, not "{type(beam)}"')
                    
                K[i,j,0]=x
                K[i,j,1]=y

    SdB = np.log10(S)*10.0
    SdB[np.isinf(SdB)] = 0
    SdB[np.isnan(SdB)] = 0
    SdB[SdB < 0] = 0
    # conf = ax.contourf(K[:,:,0], K[:,:,1], SdB, cmap=cm.plasma, vmin=0, vmax=np.max(SdB), levels=levels)
    conf = ax.pcolormesh(K[:,:,0], K[:,:,1], SdB, cmap=cm.plasma, vmin=0, vmax=np.max(SdB), shading='giraud')

    if usetex:
        ax.set_xlabel('$k_x$ [1]') #,fontsize=24)
        ax.set_ylabel('$k_y$ [1]') #,fontsize=24)
    else:
        ax.set_xlabel('kx [1]') # ,fontsize=24)
        ax.set_ylabel('ky [1]') # ,fontsize=24)
    
    plt.xticks() # fontsize=17)
    plt.yticks() # fontsize=17)
    cbar = plt.colorbar(conf, ax=ax)
    cbar.ax.set_ylabel('Gain [dB]') # ,fontsize=24)
    ax.set_title('Gain pattern') # , fontsize=24)
    ax.axis('equal')
    ax.axis('square')

    return fig, ax



def hemisphere_plot(func, plotfunc, preproc='dba',
        f_args=[], f_kw={}, p_args=[], p_kw={}, resolution=201, ax=None, vectorized=False):
    '''
    Create a hemispherical plot of some function of pointing direction 

    :param callable func: Some function that maps from a pointing vector in the
                upper hemisphere to a scalar
    :param callable plotfunc: a function with call signature like `contourf` or
                `pcolormesh`, i.e.  plotfunc(xval, yval, zval, *args, **kw)
    :param list f_args: extra arguments to `func`
    :param list f_kw  : estra keyword arguments to `func`
    :param list p_args: extra arguments to `plotfunc`
    :param list p_kw  : estra keyword arguments to `plotfunc`
    :keyword int resolution: Number of points to divide the wave vector x and y
                components into, total number of calculation points is the
                square of this number.
    :keyword plot_axis ax: Axis in which to make the plot.
                If not given, one will be created in a new figure window
    :keyword boolean vectorized: If True (default=False), then `func` accepts a
                2D array of `k` vectors, of shape (N, 3).

    :keyword: string preproc, in ['none', 'dba']
    '''

    from numpy import radians

    # We will draw a k-space circle centered on `pointing` with a radius of cos(min_elevation)
    min_elevation=0.0   # So as not to trip up code below
    pointing = np.array([0., 0., 1.])

    if ax is None:
        fh = plt.figure() # figsize=(15,7))
        ax = fh.add_subplot(111)
    else:
        fh = None

    if isinstance(plotfunc, str):
        # TODO: Some cleverness with try/except, perhaps?
        plotfunc = getattr(ax, plotfunc)



    cmin = np.cos(radians(min_elevation))

    kx = pointing[0] + np.linspace(-cmin, cmin, num=resolution)
    ky = pointing[1] + np.linspace(-cmin, cmin, num=resolution)

    K=np.zeros((resolution,resolution,2))


    if vectorized:
        K[:,:,0], K[:,:,1] = np.meshgrid(kx, ky, sparse=False, indexing='ij')
        size = resolution**2
        k = np.empty((3,size), dtype=np.float64)
        k[0,:] = K[:,:,0].reshape(1,size)
        k[1,:] = K[:,:,1].reshape(1,size)

        z2 = k[0,:]**2 + k[1,:]**2
        z2_c = (pointing[0] - k[0,:])**2 + (pointing[1] - k[1,:])**2

        inds_ = np.logical_and(z2_c < np.cos(min_elevation*np.pi/180.0)**2, z2 <= 1.0)
        not_inds_ = np.logical_not(inds_)

        k[2,inds_] = np.sqrt(1.0 - z2[inds_])
        k[2,not_inds_] = 0
        S = np.ones((1,size)) * np.nan

        S[0,inds_] = func(k[:,inds_], *f_args, **f_kw)

        S = S.reshape(resolution,resolution)

    else:
        S = np.ones((resolution,resolution)) * np.nan
        for i,x in enumerate(kx):
            for j,y in enumerate(ky):
                z2_c = (pointing[0]-x)**2 + (pointing[1]-y)**2
                z2 = x**2 + y**2
                if z2_c < np.cos(min_elevation*np.pi/180.0)**2 and z2 <= 1.0:

                    k=np.array([x, y, np.sqrt(1.0 - z2)])

                    S[i,j] = func(k, *f_args, **f_kw)

                K[i,j,0]=x
                K[i,j,1]=y

    if preproc in [None, 'none']:
        pass
    elif preproc in ['dba']:
        SdB = np.log10(S)*10.0
        SdB[np.isinf(SdB)] = 0
        SdB[np.isnan(SdB)] = 0
        # SdB[SdB < 0] = 0
        S = SdB
    else:
        print(f"preprocessor {preproc} unknown")

    # conf = ax.contourf(K[:,:,0], K[:,:,1], SdB, cmap=cm.plasma, vmin=0, vmax=np.max(SdB), levels=levels)
    # conf = ax.pcolormesh(K[:,:,0], K[:,:,1], SdB, cmap=cm.plasma, vmin=0, vmax=np.max(SdB), shading='giraud')

    hh = plotfunc(K[:,:,0], K[:,:,1], S, *p_args, **p_kw)

    ax.axis('equal')
    ax.axis('square')

    return fh, ax, hh


def new_heatmap(beam, ax=None, **kw):

    pkw = dict(cmap=cm.plasma, vmin=0, shading='giraud')

    fh, ax, hh = hemisphere_plot(beam.gain, 'pcolormesh', ax=ax, p_kw=pkw, **kw)

    usetex = True

    if usetex:
        ax.set_xlabel('$k_x$ [1]') #,fontsize=24)
        ax.set_ylabel('$k_y$ [1]') #,fontsize=24)
    else:
        ax.set_xlabel('kx [1]') # ,fontsize=24)
        ax.set_ylabel('ky [1]') # ,fontsize=24)
    
    plt.xticks() # fontsize=17)
    plt.yticks() # fontsize=17)
    cbar = plt.colorbar(hh, ax=ax)
    cbar.ax.set_ylabel('Gain [dB]') # ,fontsize=24)
    ax.set_title('Gain pattern') # , fontsize=24)

    return fh, ax, hh

