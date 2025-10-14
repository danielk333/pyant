import sys
if sys.version_info >= (3, 11):
    from typing import TypedDict, Required
else:
    from typing_extensions import Required, TypedDict

from numpy.typing import NDArray

from ..beam import Beam


class RadarStation(TypedDict, total=False):
    """The parameters available for a radar station

    Keys:
        uid: A unique string that identifies a radar station
        lat: Geographical latitude of radar station in decimal degrees  (North+).
        lon: Geographical longitude of radar station in decimal degrees (East+).
        alt: Geographical altitude above geoid surface of radar station in meter.
        beam: The radar beam for this station.
        ecef: The ITRS coordinates of the radar station.
        ecef_lat: The latitude of the ITRS coordinates of the radar station.
        ecef_lon: The longitude of the ITRS coordinates of the radar station.
        ecef_alt: The altitude of the ITRS coordinates of the radar station.
        min_elevation: The minimum elevation the radar can measure at.
        noise_temperature: The noise temperature in Kelvins intrinsic to the radar receiver.
        power: The maximum power in Watts the radar transmitter can deliver.
        power_per_element: ?
        frequency: The frequency the radar operates at.
    """

    uid: Required[str]
    transmitter: Required[bool]
    receiver: Required[bool]
    beam: Required[Beam]
    lat: Required[float]
    lon: Required[float]
    alt: Required[float]
    frequency: float
    ecef: NDArray
    ecef_lat: float
    ecef_lon: float
    ecef_alt: float
    min_elevation: float
    noise_temperature: float
    power: float
    power_per_element: float
