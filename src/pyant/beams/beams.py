"""
Handler for specific radar beam instances

"""
from ..registry import Radars, Models

RADAR_BEAMS = dict()


def avalible_beams():
    """Returns a dict listing all avalible Radars and their Models"""
    return {key: list(val.keys()) for key, val in RADAR_BEAMS.items()}


def register_radar_beam(name, model, generator, override_ok=False):
    """Registers a new radar beam."""
    if name not in RADAR_BEAMS:
        RADAR_BEAMS[name] = {}

    if model in RADAR_BEAMS[name] and not override_ok:
        raise ValueError(f"{name} with model {model} already registered")

    RADAR_BEAMS[name][model] = generator


def radar_beam_generator(name, model, override_ok=False):
    """Decorator to automatically register the radar beam generator."""

    def registrator_wrapper(generator):
        register_radar_beam(name, model, generator, override_ok=override_ok)
        return generator

    return registrator_wrapper


def beam_of_radar(radar, model, *args, **kwargs):
    """Get a predefined radar beam instance from the avalible library of beams.


    Parameters
    ----------
    radar : str or pyant.Radars
        Name or enumertation of the radar system
    model : str or pyant.Models
        Name or enumeration of the beam model
    *args : required
        Additional required positional arguments by the specific beam instance.
    **kwargs : optional
        Additional arguments supplied to the specific beam instance.

    Returns
    -------
    pyant.Beam
        A specific beam model (subclass of `Beam`) configured for the given
        radar system.

    """
    try:
        radar_item = Radars(radar)
    except ValueError:
        raise ValueError(
            f'"{radar}" radar not found. See available Radars:\n'
            + ", ".join([str(x) for x in Radars])
        )

    if radar_item not in RADAR_BEAMS:
        raise ValueError(
            f'No implemented beams for radar "{radar_item}" found. Available Radars:\n'
            + ", ".join([str(x) for x in RADAR_BEAMS])
        )

    radar_models = RADAR_BEAMS[radar_item]

    try:
        model_item = Models(model)
    except ValueError:
        raise ValueError(
            f'"{model}" model not found. See avalible Models:\n'
            + ", ".join([str(x) for x in Models])
        )

    if model_item not in radar_models:
        raise ValueError(
            f'Model "{model_item}" not implemented radar "{radar_item}" found. Available Models:\n'
            + ", ".join([str(x) for x in radar_models])
        )

    generator = radar_models[model_item]
    beam = generator(*args, **kwargs)

    return beam
