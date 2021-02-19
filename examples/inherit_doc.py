'''
Documentation inheritance
===========================
'''

from pyant import Beam, class_inherit_doc
import pyant.instances as lib

class A(Beam):
    def gain(self, k, polarization=None, ind=None):
        if len(k.shape) == 1:
            return 1.0
        else:
            return np.ones((k.shape[1],), dtype=k.dtype)

class B(Beam):
    '''Uniform beam B
    '''
    def gain(self, k, polarization=None, ind=None):
        if len(k.shape) == 1:
            return 1.0
        else:
            return np.ones((k.shape[1],), dtype=k.dtype)


@class_inherit_doc
class C(Beam):
    '''Uniform beam C
    '''
    def gain(self, k, polarization=None, ind=None):
        if len(k.shape) == 1:
            return 1.0
        else:
            return np.ones((k.shape[1],), dtype=k.dtype)



print(f'Beam docs:\n{Beam.__doc__}\n')
print(f'A docs:\n{A.__doc__}\n')
print(f'B docs:\n{B.__doc__}\n')
print(f'C docs:\n{C.__doc__}\n')