'''
Check functionality
======================
'''
import numpy as np

import pyant
import pyant.instances as lib

print(lib.__all__)

for inst in lib.__all__:
    functionality, profile = pyant.validate_functionality(getattr(lib, inst))
    print('\n'+'='*10 + f' {inst} ' + '='*10)
    for key in functionality:
        print(f'{key}:{functionality[key]}')
    for key in profile:
        print(f'{key}:{profile[key]:.2e}')

pyant.plotting.show()