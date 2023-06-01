"""
Airy disk antenna gain
========================
"""

import pyant

pyant.profile()

beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
)

# beam_c = pyant.models.Cassegrain(
#     azimuth=0,
#     elevation=90.0,
#     frequency=930e6,
#     I0=10**4.81,
#     a0=23.0,
#     a1=40.0,
# )

pyant.plotting.gain_heatmap(beam, resolution=301, min_elevation=80.0)
# pyant.plotting.gain_heatmap(beam_c, resolution=301, min_elevation=80.0)

pyant.print_profile(save_graph="example.CALLGRIND")

pyant.plotting.show()
