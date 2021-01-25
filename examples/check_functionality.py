'''
Check functionality
======================
'''
import numpy as np

import pyant
import pyant.instances as lib

print(lib.__all__)

for inst in lib.__all__:
    beam = getattr(lib, inst)
    if isinstance(beam, list):
        continue
    functionality, profile = pyant.validate_functionality(beam)

    str_len = 0
    for key in functionality:
        if len(key) > str_len:
            str_len = len(key)

    print('\n'+'='*10 + f' {inst} ' + '='*10)
    for key in functionality:
        print(f'{key.replace("_", " "):<{str_len}}:{functionality[key]}')

    print('\nProfiling:')

    str_len = 0
    for key in profile:
        if len(key) > str_len:
            str_len = len(key)

    for key in profile:
        print(f'{key.replace("_", " "):<{str_len}}:{profile[key]:.2e}')