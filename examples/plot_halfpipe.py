'''
Halfpipe radar
===========================
'''

import pyant

el = 50

beam = pyant.FiniteCylindricalParabola(
    azimuth=0,
    elevation=el,
    frequency=224.0e6,
    width=120.0,
    height=40.0,
)

# DON'T use this form!  Use azimuth instead
rotated_beam = pyant.FiniteCylindricalParabola(
    azimuth=0,
    elevation=el,
    frequency=224.0e6,
    width=120.0,
    height=40.0,
    rotation=30.0,
)

# Azimuth is angle (in degrees) clockwise
azimuthal_beam = pyant.FiniteCylindricalParabola(
    azimuth=30,
    elevation=el,
    frequency=224.0e6,
    width=120.0,
    height=40.0,
    # rotation=30.0,
)


pyant.plotting.gain_heatmap(
    beam, resolution=300, min_elevation=80.0, label='plain')
pyant.plotting.gain_heatmap(
    rotated_beam, resolution=300, min_elevation=80.0, label='rotated')
pyant.plotting.gain_heatmap(
    azimuthal_beam, resolution=300, min_elevation=80.0, label='azimuthal')
pyant.plotting.show()
