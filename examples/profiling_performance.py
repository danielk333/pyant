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

# Get the stats for only the gain method
time_g1 = next(x for x in stats if x.name == "Airy.gain")

print(f"Total time = {total1:.4f} [s]")
pyant.print_profile(stats, total=total1)

pyant.profile()

beam.sph_gain(
    azimuth=np.full((num,), 0, dtype=np.float64),
    elevation=np.full((num,), 90, dtype=np.float64),
)

stats, total2 = pyant.get_profile()
pyant.profile_stop(clear=True)

time_g2 = next(x for x in stats if x.name == "Airy.gain")

print("\n" * 2 + "USING VECTORIZED INPUTS")
print(f"Total time = {total2:.4f} [s]")
pyant.print_profile(stats, total=total2)

print("\n")
print(f"Total time speedup = {total1/total2}")
print(f"Airy.gain speedup = {time_g1.ttot/time_g2.ttot}")
