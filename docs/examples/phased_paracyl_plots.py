# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---


# # Phased paracyl plots

import numpy as np
from matplotlib import pyplot as plt

import pyant
from pyant.plotting import gain_heatmap, hemisphere_plot
from pyant.models import FiniteCylindricalParabola
from pyant.models import PhasedFiniteCylindricalParabola


def printn(*args, **kw):
    print(*args, end="", flush=True, **kw)


def mth_(text):
    if plt.rcParams["text.usetex"]:
        return f"${text}$"
    else:
        return text


def greek(name):
    if plt.rcParams["text.usetex"]:
        return mth_("\\" + name)
    else:
        return name


def phit():
    return greek("phi")


def thet():
    return greek("theta")


def get_parc():
    return PhasedFiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        phase_steering=0.0,
        depth=18.0,
        frequency=224.0e6,
        I0=None,
        width=120.0,
        height=40.0,
        degrees=True,
    )


def get_farc():
    return FiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        frequency=224.0e6,
        I0=None,
        width=120.0,
        height=40.0,
        degrees=True,
    )


def l2p_hemiplots(azim, elev, check=False):
    """
    Plot the results of local_to_pointing() for the hemisphere

    left panel(s) show phi, the off-axis angle in the azimuth direction
    right panel(s) show theta, the below-axis angle in the elevation

    Set check=1 to compare with a different (wrong!) local to pointing computation
    Set check=2 to compare angles computed using  M.dot(v) to M @ v
    """

    cmap = plt.cm.RdBu_r

    # These five definitions should be equal to the innards of local_to_pointing()
    def uvec(v):
        return v / np.linalg.norm(v, axis=0)

    Az = pyant.coordinates.rot_mat_z

    def El(t):
        return pyant.coordinates.rot_mat_x(90 - t)

    def nphi(v):
        return np.degrees(np.arcsin((El(elev) @ Az(azim) @ uvec(v))[0]))

    def nthe(v):
        return np.degrees(np.arcsin((El(elev) @ Az(azim) @ uvec(v))[1]))

    # # What happens if we normalize the projection of the rotated pointing vector before
    # # taking the sine of the x/y components?
    # # A: That is wrong. Don't do that!
    if check == 1:

        def uvx(v):
            return v[0] / np.linalg.norm(v[[0, 2]], axis=0)

        def uvy(v):
            return v[1] / np.linalg.norm(v[[1, 2]], axis=0)

        def uphi(v):
            return np.degrees(np.arcsin(uvx(El(elev).dot(Az(azim).dot(v)))))

        def uthe(v):
            return np.degrees(np.arcsin(uvy(El(elev).dot(Az(azim).dot(v)))))

    # Check instead if numpy matrix-vector product operator lets us prettify this:
    elif check == 2:

        def uphi(v):
            return np.degrees(np.arcsin(El(elev).dot(Az(azim).dot(uvec(v)))[0]))

        def uthe(v):
            return np.degrees(np.arcsin(El(elev).dot(Az(azim).dot(uvec(v)))[1]))

    levels = np.r_[-90:100:15]
    fh, ah = plt.subplots(1 + (check is not False), 2, sharex="all", sharey="all", squeeze=False)

    _, aa, ph = hemisphere_plot(
        nphi,
        "contourf",
        ax=ah[0, 0],
        preproc=None,
        p_kw=dict(levels=levels, cmap=cmap),
    )
    plt.colorbar(ph, ax=aa)
    _, aa, ph = hemisphere_plot(
        nthe,
        "contourf",
        ax=ah[0, 1],
        preproc=None,
        p_kw=dict(levels=levels, cmap=cmap),
    )
    plt.colorbar(ph, ax=aa)

    if check:
        _, aa, ph = hemisphere_plot(
            uphi,
            "contourf",
            ax=ah[1, 0],
            preproc=None,
            p_kw=dict(levels=levels, cmap=cmap),
        )
        plt.colorbar(ph, ax=aa)
        _, aa, ph = hemisphere_plot(
            uthe,
            "contourf",
            ax=ah[1, 1],
            preproc=None,
            p_kw=dict(levels=levels, cmap=cmap),
        )
        plt.colorbar(ph, ax=aa)

    ce = np.cos(np.radians(elev))
    ca, sa = np.cos(np.radians(90 - azim)), np.sin(np.radians(90 - azim))

    for ax in ah.flat:
        ax.plot(ce * ca, ce * sa, "ko")

    ah[0, 0].set_title(f"off-axis ({phit()}) M @ v")
    ah[0, 1].set_title(f"below-axis ({thet()}) M @ v")
    ah[0, 0].set_ylabel(f'{mth_("k_y")}')

    if check is False:
        ah[0, 0].set_xlabel(f'{mth_("k_x")}')
        ah[0, 1].set_xlabel(f'{mth_("k_x")}')

    else:
        ah[1, 0].set_ylabel(f'{mth_("k_y")}')
        ah[1, 0].set_xlabel(f'{mth_("k_x")}')
        ah[1, 1].set_xlabel(f'{mth_("k_x")}')

        if check == 1:
            ah[1, 0].set_title(f"off-axis ({phit()}) renormalized")
            ah[1, 1].set_title(f"below-axis ({thet()}) renormalized")
        elif check == 2:
            ah[1, 0].set_title(f"off-axis ({phit()}) M.dot(v)")
            ah[1, 1].set_title(f"below-axis ({thet()}) M.dot(v)")

    fh.suptitle(f"Azimuth {azim} Elev {elev}")


def test_local_to_pointing():
    # These four definitions should be equal to the innards of local_to_pointing()
    Az = pyant.coordinates.rot_mat_z

    def El(t):
        return pyant.coordinates.rot_mat_x(90 - t)

    # def phi(az, el, v):
    #     return np.degrees(np.arcsin(el.dot(az.dot(v))[0]))

    # def theta(az, el, v):
    #     return np.degrees(np.arcsin(el.dot(az.dot(v))[1]))

    # What happens if we normalize the projection of the rotated pointing vector before
    # taking the sine of the x/y components?
    def uvec(v):
        return v / np.linalg.norm(v, axis=0)

    def phi(az, el, v):
        return np.degrees(np.arcsin(uvec(el.dot(az.dot(v)))[0]))

    def theta(az, el, v):
        return np.degrees(np.arcsin(uvec(el.dot(az.dot(v)))[1]))

    xhat, yhat, zhat = np.eye(3)

    for elev in [0.0, 30.0, 45.0, 90.0]:
        print(f"\n *** Elevation {elev} *** ")
        printn("  ")
        for azim in [0, 90, 180, 270]:
            printn(f"az={azim:3} phi      theta  ")
        print("")

        for k, label in zip(np.eye(3), ["xhat", "yhat", "zhat"]):
            printn(label + " ")
            for azim in [0.0, 90.0, 180.0, 270.0]:
                printn(
                    f" {phi(Az(azim), El(elev), k):8.3f}  "
                    + f"{theta(Az(azim), El(elev), k):8.3f}    "
                )
            print("")


def compare(az=30, el=60, frq=60e6, with_old=False, **kw):
    parc = get_parc()

    fh, ah = plt.subplots(2 + with_old, 2, sharex="col", sharey="all")

    parc.frequency = frq
    parc.elevation = el

    parc.phase_steering = -az
    gain_heatmap(parc, ax=ah[0, 0], **kw)
    ah[0, 0].set_title(f"ph = {-az}")

    parc.phase_steering = az
    gain_heatmap(parc, ax=ah[0, 1], **kw)
    ah[0, 1].set_title(f"ph = {az}")

    parc.phase_steering = 0

    parc.azimuth = -az
    gain_heatmap(parc, ax=ah[1, 0], **kw)
    ah[1, 0].set_title(f"az = {-az}")

    parc.azimuth = az
    gain_heatmap(parc, ax=ah[1, 1])
    ah[1, 1].set_title(f"az = {az}")

    if with_old:
        farc = get_farc()
        farc.height = 40
        farc.width = 120
        farc.frequency = 30e6
        farc.elevation = 60

        farc.azimuth = -az
        gain_heatmap(farc, ax=ah[2, 0])
        ah[2, 0].set_title(f"(unphaseable) az = {-az}")

        farc.azimuth = az
        gain_heatmap(farc, ax=ah[2, 1])
        ah[2, 1].set_title(f"(unphaseable) az = {az}")

        # ah[2,0].set_xlim([-0.2, 0.8])

    ah[0, 0].set_ylim([-0.2, 0.8])
    ah[0, 0].set_xlim([-0.8, 0.2])
    ah[0, 1].set_xlim([-0.2, 0.8])

    fh.suptitle(f"el={el}, frq={frq/1e6:.4g} MHz")
    plt.show()


az, el = 15.0, 70.0
l2p_hemiplots(azim=az, elev=el)
l2p_hemiplots(azim=az, elev=el, check=1)
l2p_hemiplots(azim=az, elev=el, check=2)
compare(az=az, el=el, frq=120e6, resolution=401)
