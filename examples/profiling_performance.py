"""
Profiling
==========
"""
import numpy as np
import pyant

beam = pyant.models.Airy(
    azimuth=0,
    elevation=90.0,
    frequency=930e6,
    I0=10**4.81,
    radius=23.0,
    degrees=True,
)

num = 10000

# Does not work if yappi is not installed
pyant.profile()

for ind in range(num):
    beam.sph_gain(azimuth=0, elevation=90)

stats, total1 = pyant.get_profile()
pyant.profile_stop(clear=True)

print(f"Total time = {total1:.2f} [s]")
pyant.print_profile(stats, total=total1)

pyant.profile()

beam.sph_gain(
    azimuth=np.full((num,), 0, dtype=np.float64),
    elevation=np.full((num,), 90, dtype=np.float64),
)

stats, total2 = pyant.get_profile()
pyant.profile_stop(clear=True)

print("\n" * 2)
print(f"Total time = {total2:.2f} [s]")
pyant.print_profile(stats, total=total2)

print(f"Speedup = {total1/total2}")
