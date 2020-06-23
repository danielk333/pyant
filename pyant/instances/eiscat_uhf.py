#!/usr/bin/env python

'''

'''



# def uhf_meas(k_in,beam):
#     '''Measured UHF beam pattern

#     '''
#     theta = coord.angle_deg(beam.on_axis,k_in)
#     # scale beam width by frequency
#     sf=beam.f/930e6
    
#     return(beam.I_0*beam.gf(sf*np.abs(theta)))

# def uhf_beam(az0, el0, I_0, f, beam_name='UHF Measured beam'):
#     '''# TODO: Description.

#     '''
#     beam = antenna.BeamPattern(uhf_meas, az0, el0, I_0, f, beam_name=beam_name)

#     bmod=np.genfromtxt("data/bp.txt")
#     angle=bmod[:,0]
#     gain=10**(bmod[:,1]/10.0)
#     gf=sio.interp1d(np.abs(angle),gain)
    
#     beam.gf = gf
#     return beam

