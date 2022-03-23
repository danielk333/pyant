'''
Predefined instances
======================
'''
import pyant

all_beams = pyant.avalible_beams()

for radar in all_beams:
    for model in all_beams[radar]:
        print(f'plotting {model} model of {radar} radar')
        beam = pyant.beam_of_radar(radar, model)
        fig, ax, _ = pyant.plotting.gain_heatmap(
            beam, resolution=100, min_elevation=45.0
        )
        ax.set_title(f'{radar} radar - {model} model')

pyant.plotting.show()
