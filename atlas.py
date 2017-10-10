from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pdb
#import pidly

# Overview:
#
# TME 26 Aug 2016
# These limb darkening routines should be made into proper modular objects.
# They're already set up in a way that should make this fairly straightforward.
# This is on the to-do list.
#
# TME 13 Jan 2014
#
# Before using the routines in this module, you need to make sure that you
# have the appropriate input files downloaded from Robert Kurucz's website.
# To do this, go to:
#
#   http://kurucz.harvard.edu
#
# and navigate to the appropriate file. For the angular intensities, first
# on "Grids of model atmospheres" and then select the appropriate category
# of model. These have names of the form "GRIDxyyz" where x is either "P"
# for + or "M" for -, yy is the metallicity [M/H] (or [Fe/H]), and z is any
# additional suffix for further subdivisions. For example, "GRIDP03" will take
# you to the models with [M/H]=+0.3, whereas GRIDM30 take you to the models
# with [M/H]=-3.0. Note that between the two yy digits, it is assumed there is
# a decimal point. Once you're onto the next page, there will be a list of
# files, many of which are quite large. The ones that give emergent intensity
# as a function of mu=cos(theta), where theta is the angle between the line of
# sight and the emergent intensity, have names of the form i*.pck, i.e. an "i"
# prefix - see previous page you were just on for further details. Once you've
# decided which file you'd like to download, you can right click on it and
# choose "Save link as...", but I prefer curl (or wget), using something like:
#
#  >> curl -O http://kurucz.harvard.edu/grids/gridP03/ip03k2.pck
#
# Then you have all you need to start estimating limb darkening coefficients
# with the routines below.
#
# A basic calling sequence for the routines in this module would be:
#
#   1>> mu, wav, intens = atlas.read_grid( model_filepath='im01k2new.pck', \
#                                          teff=6000, logg=4., new_grid=False )
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



def read_grid( model_filepath=None, teff=None, logg=None, new_grid=False ):
    """
    Given the full path to an ATLAS model grid, along with values for
    Teff and logg, this routine extracts the values for the specific
    intensity as a function of mu=cos(theta), where theta is the angle
    between the line of site and the emergent radiation. Calling is:

      mu, wav, intensity = atlas.read_grid( model_filepath='filename.pck', \
                                            teff=6000, logg=4.5, vturb=2. )

    Note that the input grids correspond to a given metallicity and
    vturb parameter. So those parameters are controlled by defining
    the model_filepath input appropriately.

    The units of the output variables are:
      mu - unitless
      wav - nm
      intensity - erg/cm**2/s/nm/ster

    Another point to make is that there are some minor issues with the
    formatting of 'new' ATLAS  grids on the Kurucz website. This
    routine will fail on those if you simply download them and feed
    them as input, unchanged. This is because:
      - They have an extra blank line at the start of the file.
      - More troublesome, the last four wavelengths of each grid
        are printed on a single line, which screws up the expected
        structure that this routine requires to read in the file.
    This is 

    """

    # Row dimensions of the input file:
    if new_grid==False:
        nskip = 0 # number of lines to skip at start of file
        nhead = 3 # number of header lines for each grid point
        nwav = 1221 # number of wavelengths for each grid point
    else:
        nskip = 0 # number of lines to skip at start of file
        nhead = 4 # number of header lines for each grid point
        nwav = 1216 # number of wavelengths for each grid point
    nang = 17 # number of angles for each grid point
    # Note: The 'new' model grids don't quite have the 
    # same format, so they won't work for this code.

    print( '\nReading in the model grid...' )
    ifile = open( model_filepath, 'rU' )
    ifile.seek( 0 )
    rows = ifile.readlines()
    ifile.close()
    rows = rows[nskip:]
    nrows = len( rows )
    print( 'Done.' )

    # The angles, where mu=cos(theta):
    mus = np.array( rows[nskip+nhead-1].split(), dtype=float )

    # Read in the teff, logg and vturb values
    # for each of the grid points:
    row_ixs = np.arange( nrows )
    header_ixs = row_ixs[ row_ixs%( nhead + nwav )==0 ]
    if new_grid==True:
        header_ixs += 1
        header_ixs = header_ixs[:-1]
    ngrid = len( header_ixs )
    teff_grid = np.zeros( ngrid )
    logg_grid = np.zeros( ngrid )
    for i in range( ngrid ):
        header = rows[header_ixs[i]].split()
        teff_grid[i] = float( header[1] )
        logg_grid[i] = header[3]

    # Identify the grid point of interest:
    logg_ixs = ( logg_grid==logg )
    teff_ixs = ( teff_grid==teff )

    # Extract the intensities at each of the wavelengths
    # as a function of wavelength:
    grid_ix = ( logg_ixs*teff_ixs )
    row_ix = int( header_ixs[grid_ix] )
    grid_lines = rows[row_ix+nhead:row_ix+nhead+nwav]
    grid = []
    for i in range( nwav ):
        grid += [ grid_lines[i].split() ]
    if new_grid==True:
        grid=grid[:-1]
    grid = np.array( np.vstack( grid ), dtype=float )
    wavs_nm = grid[:,0]
    intensities = grid[:,1:]

    nmus = len( mus )
    for i in range( 1, nmus ):
        intensities[:,i] = intensities[:,i]*intensities[:,0]/100000.

    # Convert the intensities from per unit frequency to
    # per nm in wavelength:
    for i in range( nmus ):
        intensities[:,i] /= ( wavs_nm**2. )

    
    return mus, wavs_nm, intensities
