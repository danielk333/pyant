'''
Predefined parameters
======================
'''
from pprint import pprint

import pyant

pprint(pyant.avalible_radar_info())

info = pyant.parameters_of_radar('eiscat_uhf')
print('\nPredefined parameters for "eiscat_uhf"')
for key in info:
    print(f'{key}: {info[key]:.2e} {pyant.UNITS[key]}')
