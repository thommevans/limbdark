import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import os, sys, pdb

# 2nov2012 TME
#
# Before using these routines, you need to make
# sure that you have the appropriate input files
# downloaded from Robert Kurucz's website. To
# do this, go to:
#
#   http://kurucz.harvard.edu
#
# and navigate to the appropriate file. For the
# angular intensities, first go to "Grids of model
# atmospheres" and then select the appropriate
# category of model. These have names of the form
# "GRIDxyyz" where x is either "P" for + or "M"
# for -, yy is the metallicity [M/H] (or [Fe/H]),
# and z is any additional suffix for further
# subdivisions. For example, "GRIDP03" will take
# you to the models with [M/H]=+0.3, whereas
# GRIDM30 take you to the models with [M/H]=-3.0.
# Note that between the two yy digits, it is
# assumed there is a decimal point. Once you're
# onto the next page, there will be a list of
# files, many of which are quite large. The ones
# that give emergent intensity as a function of
# mu=cos(theta), where theta is the angle between
# the line of sight and the emergent intensity,
# have names of the form i*.pck, i.e. an "i"
# prefix - see previous page you were just on
# for further details. Once you've decided which
# file you'd like to download, you can right
# click on it and choose "Save link as...", but
# I prefer curl (or wget), using something like:
#
#  >> curl -O http://kurucz.harvard.edu/grids/gridP03/ip03k2.pck
#
# Then you have all you need to start estimating
# limb darkening coefficients with the routines
# below.
#
# A basic calling sequence for the routines in this
# module would be:
# 
#   >> em_mus, em_wavs, ems = kurucz.readgrid( model_filepath='im01k2new.pck', \
#                                              teff=6250, \
#                                              logg=4.5, \
#                                              nskip=1, \
#                                              nhead=3, \
#                                              nwav=1221, \
#                                              nang=17 )
#   >> tr_curve = np.loadtxt( tr_filename )
#   >> tr_wavs = tr_curve[:,0]
#   >> tr_vals = tr_curve[:,1]
#   >> coeffs = kurucz.fit_law( em_mus, em_wavs, ems, 
#                               tr_wavs, tr_vals, ld_law='nonlin' )
#
# NOTE:
# 1. I found a frustrating amount of heterogeneity between the format
#    of some of the input files on Kurucz's website. For this reason,
#    I think the best option is to manually determine the format of
#    the input file you're interested in at any given time, and pass
#    this into the read_grid() routine in the form of the variables
#    nskip, nhead, nwav and nang. Typical values I've encountered so
#    far are:
#      nskip = 1
#      nhead = 3
#      nwav = 1221
#      nang = 17
#    but annoyingly these can't be frozen as global variables in this
#    module, because I've also encountered cases where nwav = 1217.
# 2. Another issue I've had with the files on Kurucz's website is that
#    the values in the column for mu=1.0 are all pretty close to zero,
#    which just seems wrong when compared with values at the other mus.
#    For this reason, I've been manually excluding them from the input
#    that I pass into the fit_law() routine, after having read it in
#    using the read_grid() routine.
# 3. Following Sing (2010), it might also be a good idea to exclude
#    values mu<0.05 when doing the fits, particulary for the crude
#    limb darkening laws like linear and quadratic. I think this is
#    because these points can bias the fits the most, and the crude
#    models do a poor job of reproducing those points.
#


def read_grid( model_filepath=None, teff=None, logg=None, nskip=1, nhead=3, nwav=1221, nang=17 ):
    """
    Given the full path to a Kurucz input file, along with
    values for Teff and logg, extracts the values for the
    emergent intensity as a function of mu=cos(theta) where
    theta is the angle between the line of site and the
    emergent radiation.
    """

    ifile = open( model_filepath, 'rU' )
    rows = ifile.readlines()
    ifile.close()

    n = len( rows )
    found = False
    k = 0
    while found==False:
        cols = rows[ nskip + k*( nhead + nwav ) ].split()
        if ( float( cols[1] )==teff )*\
           ( float( cols[3] )==logg ):
            found = True
        else:
            k += 1

    if found==True:
        ix0 = nskip + k*( nhead + nwav ) + nhead
        em_mus = np.array( rows[ ix0 - 1 ].split(), dtype=float )
        print '\nFormat of output:\n'
        for i in range( nhead ):
            print rows[ ix0 - nhead + i ]
        print ''
        block = np.zeros( [ 1, 1+nang ] )
        for j in range( nwav ):
            try:
                block = np.vstack( [ block, rows[ix0+j].split() ] )
            except:
                pass
        block = np.array( block, dtype=float )
        em_wavs = block[:,0]
        ems = block[:,1:]
    else:
        print '\n\nCould not match for TEFF={0} and LOGG={1}'.format( teff, logg )
        em_mus = None
        block = None
        
    return em_mus, em_wavs, ems
        
    
def fit_law( em_mus, em_wavs, ems, tr_wavs, tr_vals, ld_law='nonlin' ):
    """
    Given model values for the stellar emission as a function 
    of wavelength and viewing angle (i.e. mu=cos(theta) ),
    routine will use linear least squares to solve for limb
    darkening coefficients.

    INPUTS:
      em_mus - Nx0 array containing the viewing angles that
               the model has been evaluated for.
      em_wavs - Mx0 array containing the individual wavelengths
                that the model has been evaluated for, at each
                of the em_mus values.
      ems - NxM array containing the model emission values at
            each point on the grid spanned by viewing angle and
            wavelength. The units are not important here,
            because the coefficients found by this routine using
            linear least squares are normalised appropriately
            before being returned as output.
      tr_wavs - Kx0 array containing the wavelengths of the
                transmission curve.

    NOTE:
     1. Make sure em_wavs and tr_wavs have the same units.
     2. The tr_vals array does not need to be normalised
        as this will be done anyway inside this routine.
    """

    # We at least expect to have the transmission
    # curve well-characterised, so it is sensible
    # to interpolate over it to the points where
    # the stellar model is evaluated at.

    # If we have a finite-width bandpass, we will
    # calculate the relative weights at each wavelength
    # using interpolation:
    if np.rank( tr_wavs )>0:

        ixs = ( em_wavs>tr_wavs.min() )*( em_wavs<tr_wavs.max() )
        if len( np.unique( tr_vals ) )>1:
            interpfunc = scipy.interpolate.interp1d( tr_wavs, tr_vals, kind='linear' )
            weights = interpfunc( em_wavs[ixs] )
        else:
            weights = np.ones( len( em_wavs[ixs] ) )
        em_integ = np.sum( ems[ixs,:].T*weights, axis=1 )
    # Otherwise, if we have an artificial situation
    # where we're interested in a single wavelength,
    # we will interpolate between the two closest
    # wavelengths for which the stellar model has
    # been evaluated:
    else:
        nem = len( em_wavs )
        under = em_wavs<tr_wavs
        above = em_wavs>tr_wavs
        em_integ = 0.5*( ems[under,:][-1,:] + ems[above,:][0,:] )

    em_integ /= em_integ.max()
    
    # Prepare the linear basis matrix depending on the
    # limb darkening law being solved for:
    if ld_law=='nonlin':
        phi = claret2004_nonlin_ld( em_mus, coeffs=None )
    elif ld_law=='quad':
        phi = quadratic_ld( em_mus, coeffs=None )
    elif ld_law=='lin':
        phi = linear_ld( em_mus, coeffs=None )
    else:
        pdb.set_trace() # haven't added any others yet

    # Find the coefficients by linear least squares,
    # then ensure the coefficients have been normalised
    # appropriately before returning them as output:
    coeffs = np.linalg.lstsq( phi, em_integ )[0]
    coeffs /= coeffs[0]

    # Exclude the unity term:
    coeffs = coeffs[1:]
    
    return coeffs


def linear_ld( mus, coeffs=None ):
    """
    Linear limb darkening law.

    I(mu) = c0 - c1*( 1-mu )

    Note, that if coeffs==None, then the basis
    matrix will be returned in preparation for
    finding the limb darkening coeffecients by
    linear least squares. Otherwise, coeffs
    should be an array with 1 entries, one for
    each of the linear limb darkening
    coefficients, in which case the output will
    be the limb darkening law evaluated with
    those coefficients at the locations of the
    mus entries.
    """

    if coeffs==None:
        phi = np.ones( [ len( mus ), 2 ] )
        phi[:,1] = -( 1.0 - mus )

    else:
        phi = coeffs[0] - coeffs[1]*( 1.0 - mus )

    return phi


def quadratic_ld( mus, coeffs=None ):
    """
    Quadratic limb darkening law.

    I(mu) = c0 - c1*( 1-mu ) - c2*( ( 1-mu )**2. )

    Note, that if coeffs==None, then the basis
    matrix will be returned in preparation for
    finding the limb darkening coeffecients by
    linear least squares. Otherwise, coeffs
    should be an array with 2 entries, one for
    each of the quadratic limb darkening
    coefficients, in which case the output will
    be the limb darkening law evaluated with
    those coefficients at the locations of the
    mus entries.
    """

    if coeffs==None:
        phi = np.ones( [ len( mus ), 3 ] )
        phi[:,1] = -( 1.0 - mus )
        phi[:,2] = -( ( 1.0 - mus )**2. )

    else:
        phi = coeffs[0] - coeffs[1]*( 1.0 - mus ) \
              - coeffs[2]*( ( 1.0 - mus )**2. )

    return phi


def claret2004_nonlin_ld( mus, coeffs=None ):
    """
    The nonlinear limb darkening law as defined
    in Equation 5 of Claret et al 2004.

    Note, that if coeffs==None, then the basis
    matrix will be returned in preparation for
    finding the limb darkening coeffecients by
    linear least squares. Otherwise, coeffs
    should be an array with 4 entries, one for
    each of the nonlinear limb darkening
    coefficients, in which case the output will
    be the limb darkening law evaluated with
    those coefficients at the locations of the
    mus entries.
    """

    if coeffs==None:
        phi = np.ones( [ len( mus ), 5 ] )
        phi[:,1] = - ( 1.0 - mus**(1./2.) )
        phi[:,2] = - ( 1.0 - mus )
        phi[:,3] = - ( 1.0 - mus**(3./2.) )
        phi[:,4] = - ( 1.0 - mus**(2.) )

    else:
        phi = coeffs[0] - coeffs[1] * ( 1.0 - mus**(1./2.) ) \
              - coeffs[2] * ( 1.0 - mus ) \
              - coeffs[3] * ( 1.0 - mus**(3./2.) ) \
              - coeffs[4] * ( 1.0 - mus**(2.) )

    return phi
