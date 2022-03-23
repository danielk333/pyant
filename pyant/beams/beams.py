'''
Handler for specific radar beam instances

'''
RADAR_BEAMS = dict()


def avalible_beams():
    '''Returns a dict listing all avalible radars and their models
    '''
    return {key: list(val.keys()) for key, val in RADAR_BEAMS.items()}


def register_radar_beam(name, model, generator, override_ok=False):
    '''Registers a new radar beam.
    '''
    if name not in RADAR_BEAMS:
        RADAR_BEAMS[name] = {}

    if model in RADAR_BEAMS[name] and not override_ok:
        raise ValueError(f'{name} with model {model} already registered')

    RADAR_BEAMS[name][model] = generator


def radar_beam_generator(name, model, override_ok=False):
    '''Decorator to automatically register the radar beam generator.
    '''
    def registrator_wrapper(generator):
        register_radar_beam(name, model, generator, override_ok=override_ok)
        return generator
    return registrator_wrapper


def beam_of_radar(name, model, **kwargs):
    '''Get a predefined radar beam instance from the avalible library of beams.
    '''
    if name not in RADAR_BEAMS:
        raise ValueError(
            f'"{name}" beam not found. See avalible beams:\n'
            + ', '.join(RADAR_BEAMS.keys())
        )
    radar = RADAR_BEAMS[name]
    if model not in radar:
        raise ValueError(
            f'"{model}" model for {name} not found. See avalible models:\n'
            + ', '.join(radar.keys())
        )
    generator = RADAR_BEAMS[name][model]
    beam = generator(**kwargs)

    return beam
