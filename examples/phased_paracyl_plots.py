
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt

import pyant
from pyant.plotting import gain_heatmap


def compare(with_old=False):
    pparc = pyant.instances.tsdr_phased

    fh, ah = plt.subplots(2+with_old,2, sharex='col', sharey='all')


    pparc.frequency = 30e6
    pparc.elevation = 60


    pparc.phase_steering = -30
    gain_heatmap(pparc, ax=ah[0,0])
    ah[0,0].set_title('ph = -30')

    pparc.phase_steering = 30
    gain_heatmap(pparc, ax=ah[0,1])
    ah[0,1].set_title('ph = 30')

    pparc.phase_steering = 0

    pparc.azimuth = -30
    gain_heatmap(pparc, ax=ah[1,0])
    ah[1,0].set_title('az = -30')

    pparc.azimuth = 30
    gain_heatmap(pparc, ax=ah[1,1])
    ah[1,1].set_title('az = 30')


    if with_old:
        parc = pyant.instances.tsdr
        parc.width = 120
        parc.frequency = 30e6
        parc.elevation = 60

        parc.azimuth = -30
        gain_heatmap(parc, ax=ah[2,0])
        ah[2,0].set_title('az = -30')

        parc.azimuth = 30
        gain_heatmap(parc, ax=ah[2,1])
        ah[2,1].set_title('az = 30')

        # ah[2,0].set_xlim([-0.2, 0.8])

    ah[0,0].set_ylim([-0.2, 0.8])
    ah[0,0].set_xlim([-0.8, 0.2])
    ah[0,1].set_xlim([-0.2, 0.8])


    fh.suptitle('el=60')
    plt.show()



