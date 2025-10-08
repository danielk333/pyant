from ..models import Gaussian

def generate_tsdr(sub_modules=1, ):
    """Tromso Space Debris Radar system with all panels moving as a whole [1]_.

    Notes
    -----
    Has an extra method called :code:`calibrate` that numerically calculates
    the integral of the gain and scales the gain pattern according.



    """
    return FiniteCylindricalParabola(
        azimuth=0,
        elevation=90.0,
        frequency=224e6,
        I0=None,
        width=120.0,
        height=40.0,
        degrees=True,
    )
