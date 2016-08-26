import pdb, sys
import numpy as np
import idlsave
import matplotlib.pyplot as plt

# Overview:
#
# TME 26 Aug 2016
# These limb darkening routines should be made into proper modular objects.
# They're already set up in a way that should make this fairly straightforward.
# This is on the to-do list.
#
# A basic calling sequence for the routines in this module would be:
#
#   1>> mu, wav, intens = stagger.read_grid( model_filepath='im01k2new.pck', \
#                                            teff=6000, logg=4. )
#   2>> tr_curve = np.loadtxt( tr_filename )
#   3>> tr_wavs = tr_curve[:,0] # convert these wavelengths to nm if necessary
#   4>> tr_vals = tr_curve[:,1]
#   5>> ld_coeffs = ld.fit_law( mu, wav, intens, tr_wavs, \
#                               cuton_wav_nm=800, cutoff_wav_nm=1000, \
#                               passband_sensitivity=tr_vals, plot_fits=True )
#
# Stepping through each of the above commands:
#   1>> Reads in the model grid. Note that teff and logg have to correspond
#       to actual points on the grid - no interpolation is performed. The
#       'new_grid' flag refers to whether or not the grid is one of the new
#       ones, which would be indicated in the filename. There seems to have
#       been a few little hiccups with the formatted of these new grids, so
#       the routine has to account for these when it reads them in. The output
#       is then an array for the mu values, an array for the wavelength values
#       in units of nm, and an array for the intensities at each point spanned
#       by the mu-wavelength grid.
#   2>> Reads in an external file containing the passband transmission function
#       as a function of wavelength.
#   3>> Unpacks the wavelengths of the passband, which should be in nm.
#   4>> Unpacks the relative transmission of the passband at the different
#       wavelengths.
#   5>> Evaluates the limb darkening parameters for four separate laws -
#       linear, quadratic, three-parameter nonlinear and four-parameter nonlinear.
#       Note that the keyword argument 'passband_sensitivty' can be set to None
#       if you want a simple boxcar transmission function.
#


def read_grid( model_filepath=None ):
    z = idlsave.read( model_filepath )['mmd']
    mus = np.array( z['mu']  )
    wavs_A_in = z['lam']
    intensities_in = z['flx']    
    nang = len( mus )
    nwav = len( wavs_A_in[0] )
    wavs_nm = np.zeros( [nwav,nang] )
    intensities = np.zeros( [nwav,nang] )
    for i in range( nang ):
        wavs_nm[:,i] = wavs_A_in[i]/10.
        intensities[:,i] = intensities_in[i]
    return mus, wavs_nm, intensities
