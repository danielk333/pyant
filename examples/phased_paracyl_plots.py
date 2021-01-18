
import numpy as np
import scipy as sp

from numpy import degrees, radians

from matplotlib import pyplot as plt

import pyant
from pyant.plotting import gain_heatmap, hemisphere_plot

from pyant import FiniteCylindricalParabola
from pyant import PhasedFiniteCylindricalParabola

def printn(*args, **kw):
    print(*args, end="", flush=True, **kw)

def mth_(text):
    if plt.rcParams['text.usetex']:
        return f'${text}$'
    else:
        return text

def greek(name):
    if plt.rcParams['text.usetex']:
        return mth_('\\' + name)
    else:
        return name

def phit(): return greek('phi')
def thet(): return greek('theta')

def get_parc():
    return PhasedFiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        phase_steering = 0.0,
        depth=18.0,
        frequency=224.0e6,
        I0=None,
        width=120.0,
        height=40.0,
    )

def get_farc():
    return FiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        frequency=224.0e6,
        I0=None,
        width=120.0,
        height=40.0,
    )


def l2p_hemiplots(azim, elev):

    cmap = plt.cm.RdBu_r

    # These five definitions should be equal to the innards of local_to_pointing()
    def uvec(v): return v/np.linalg.norm(v, axis=0)
    Az = pyant.coordinates.rot_mat_z
    def El(t): return pyant.coordinates.rot_mat_x(90-t)

    def uphi(v): return np.degrees(np.arcsin(El(elev).dot(Az(azim).dot(uvec(v)))[0]))
    def uthe(v): return np.degrees(np.arcsin(El(elev).dot(Az(azim).dot(uvec(v)))[1]))

    # # What happens if we normalize the projection of the rotated pointing vector before
    # # taking the sine of the x/y components?
    # # A: That is wrong. Don't do that!
    # def uvx(v): return v[0]/np.linalg.norm(v[[0,2]], axis=0)
    # def uvy(v): return v[1]/np.linalg.norm(v[[1,2]], axis=0)
    #
    # def nphi(v): return np.degrees(np.arcsin(uvx(El(elev).dot(Az(azim).dot(v)))))
    # def nthe(v): return np.degrees(np.arcsin(uvy(El(elev).dot(Az(azim).dot(v)))))

    # Check instead if numpy matrix-vector product operator lets us prettify this:

    def nphi(v): return np.degrees(np.arcsin((El(elev) @ Az(azim) @ uvec(v))[0]))
    def nthe(v): return np.degrees(np.arcsin((El(elev) @ Az(azim) @ uvec(v))[1]))

    levels = np.r_[-90:100:15]
    fh, ah = plt.subplots(2, 2, sharex='all', sharey='all')

    _, aa, ph = hemisphere_plot(uphi, 'contourf', ax=ah[0,0], preproc=None,
                               vectorized=True, p_kw=dict(levels=levels, cmap=cmap))
    plt.colorbar(ph, ax=aa)
    _, aa, ph = hemisphere_plot(uthe, 'contourf', ax=ah[0,1], preproc=None,
                                vectorized=True, p_kw=dict(levels=levels, cmap=cmap))
    plt.colorbar(ph, ax=aa)
    _, aa, ph = hemisphere_plot(nphi, 'contourf', ax=ah[1,0], preproc=None,
                                vectorized=True, p_kw=dict(levels=levels, cmap=cmap))
    plt.colorbar(ph, ax=aa)
    _, aa, ph = hemisphere_plot(nthe, 'contourf', ax=ah[1,1], preproc=None,
                                vectorized=True, p_kw=dict(levels=levels, cmap=cmap))
    plt.colorbar(ph, ax=aa)

    ce = np.cos(np.radians(elev))
    ca, sa = np.cos(np.radians(90-azim)), np.sin(np.radians(90-azim))

    for ax in ah.flat:
        ax.plot(ce*ca, ce*sa, 'ko')

    ah[0,0].set_title(f'off-axis ({phit()}) M.dot(v)')
    ah[1,0].set_title(f'off-axis ({phit()}) M @ v')

    ah[0,1].set_title(f'below-axis ({thet()}) M.dot(v)')
    ah[1,1].set_title(f'below-axis ({thet()}) M @ v')

    ah[0,0].set_ylabel(f'{mth_("k_y")}')
    ah[1,0].set_ylabel(f'{mth_("k_y")}')

    ah[1,0].set_xlabel(f'{mth_("k_x")}')
    ah[1,1].set_xlabel(f'{mth_("k_x")}')

    fh.suptitle(f'Azimuth {azim} Elev {elev}')







def test_local_to_pointing():

    # These four definitions should be equal to the innards of local_to_pointing()
    Az = pyant.coordinates.rot_mat_z
    def El(t): return pyant.coordinates.rot_mat_x(90-t)

    def phi(az, el, v): return np.degrees(np.arcsin(el.dot(az.dot(v))[0]))
    def theta(az, el, v): return np.degrees(np.arcsin(el.dot(az.dot(v))[1]))

    # What happens if we normalize the projection of the rotated pointing vector before
    # taking the sine of the x/y components?
    def uvec(v): return v/np.linalg.norm(v, axis=0)

    def phi(az, el, v): return np.degrees(np.arcsin(uvec(el.dot(az.dot(v)))[0]))
    def theta(az, el, v): return np.degrees(np.arcsin(uvec(el.dot(az.dot(v)))[1]))


    xhat, yhat, zhat = np.eye(3)

    for elev in [0., 30., 45., 90.]:
        print(f"\n *** Elevation {elev} *** ")
        printn("  ")
        for azim in [0, 90, 180, 270]:
            printn(f"az={azim:3} phi      theta  ")
        print("")

        for k, label in zip(np.eye(3), ['xhat', 'yhat', 'zhat']):
            printn(label + " ")
            for azim in [0., 90., 180., 270.]:
                printn(f" {phi(Az(azim), El(elev), k):8.3f}  " \
                        + f"{theta(Az(azim), El(elev), k):8.3f}    ")
            print("")



def compare(az=30, el=60, frq=60e6, with_old=False):

    parc = get_parc()

    fh, ah = plt.subplots(2+with_old,2, sharex='col', sharey='all')


    parc.frequency = frq
    parc.elevation = el


    parc.phase_steering = -az
    gain_heatmap(parc, ax=ah[0,0])
    ah[0,0].set_title(f'ph = {-az}')

    parc.phase_steering = az
    gain_heatmap(parc, ax=ah[0,1])
    ah[0,1].set_title(f'ph = {az}')

    parc.phase_steering = 0

    parc.azimuth = -30
    gain_heatmap(parc, ax=ah[1,0])
    ah[1,0].set_title(f'az = {-az}')

    parc.azimuth = 30
    gain_heatmap(parc, ax=ah[1,1])
    ah[1,1].set_title(f'az = {az}')


    if with_old:
        farc = get_farc()
        farc.height = 40
        farc.width = 120
        farc.frequency = 30e6
        farc.elevation = 60

        farc.azimuth = -30
        gain_heatmap(farc, ax=ah[2,0])
        ah[2,0].set_title('(unphaseable) az = -30')

        farc.azimuth = 30
        gain_heatmap(farc, ax=ah[2,1])
        ah[2,1].set_title('(unphaseable) az = 30')

        # ah[2,0].set_xlim([-0.2, 0.8])

    ah[0,0].set_ylim([-0.2, 0.8])
    ah[0,0].set_xlim([-0.8, 0.2])
    ah[0,1].set_xlim([-0.2, 0.8])


    fh.suptitle(f'el={el}, frq={frq/1e6:.4g} MHz')
    plt.show()



