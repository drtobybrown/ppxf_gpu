"""
###############################################################################

    Copyright (C) 2001-2024, Michele Cappellari
    E-mail: michele.cappellari_at_physics.ox.ac.uk

    This software is provided as is without any warranty whatsoever.
    Permission to use, for non-commercial purposes is granted.
    Permission to modify for personal or internal use is granted,
    provided this copyright and disclaimer are included unchanged
    at the beginning of the file. All other rights are reserved.

###############################################################################

    This file contains the following independent programs:

    1) log_rebin() to rebin a spectrum logarithmically
    2) determine_goodpixels() to mask gas emission lines for pPXF
    3) determine_mask() to mask gas emission lines for pPXF
    4) vac_to_air() to convert vacuum to air wavelengths
    5) air_to_vac() to convert air to vacuum wavelengths
    6) emission_lines() to create gas emission line templates for pPXF
    7) gaussian_filter1d() **deprecated** to convolve a spectrum with a variable sigma
    8) plot_weights_2d() to plot an image of the 2-dim weights
    9) convolve_gauss_hermite() to accurately convolve a spectrum with a LOSVD
    10) synthetic_photometry() to compute photometry from spectra and filters
    11) mag_sun() to compute the Sun absolute magnitude in any band
    12) mag_spectrum() to compute the apparent magnitude from a spectrum in any band
    13) varsmooth() to convolve a spectrum with a variable sigma using FFT

"""
from importlib import resources

import numpy as np
import matplotlib.pyplot as plt

from ppxf.ppxf import losvd_rfft, rebin

###############################################################################
#
# NAME:
#   log_rebin
#
# MODIFICATION HISTORY:
#   V1.0.0: Using interpolation. Michele Cappellari, Leiden, 22 October 2001
#   V2.0.0: Analytic flux conservation. MC, Potsdam, 15 June 2003
#   V2.1.0: Allow a velocity scale to be specified by the user.
#       MC, Leiden, 2 August 2003
#   V2.2.0: Output the optional logarithmically spaced wavelength at the
#       geometric mean of the wavelength at the border of each pixel.
#       Thanks to Jesus Falcon-Barroso. MC, Leiden, 5 November 2003
#   V2.2.1: Verify that lam_range[0] < lam_range[1].
#       MC, Vicenza, 29 December 2004
#   V2.2.2: Modified the documentation after feedback from James Price.
#       MC, Oxford, 21 October 2010
#   V2.3.0: By default, now preserve the shape of the spectrum, not the
#       total flux. This seems what most users expect from the procedure.
#       Set the keyword /FLUX to preserve flux like in previous versions.
#       MC, Oxford, 30 November 2011
#   V3.0.0: Translated from IDL into Python. MC, Santiago, 23 November 2013
#   V3.1.0: Fully vectorized log_rebin. Typical speed up by two orders of magnitude.
#       MC, Oxford, 4 March 2014
#   V3.1.1: Updated documentation. MC, Oxford, 16 August 2016
#   V3.2.0: Support log-rebinning of arrays of spectra. MC, Oxford, 27 May 2021
#   V4.0.0: Support log-rebinning with non-uniform wavelength sampling.
#       MC, Oxford, 13 April 2022
#   V4.0.1: Fixed program stop with non-uniform log-rebinning of arrays.
#       MC, Oxford, 16 June 2022
#   V4.0.2: Make `velscale` a scalar. MC, Oxford, 4 January 2023


def log_rebin(lam, spec, velscale=None, oversample=1, flux=False):
    """Logarithmically rebin a spectrum or an array of spectra.

    This function logarithmically rebins a spectrum, or the first dimension of
    an array of spectra, while rigorously conserving flux. The photons in the
    spectrum are redistributed to a new grid of pixels with logarithmic
    sampling in the spectral direction.

    The function can operate in two modes based on the `flux` parameter.
    When `flux=True`, it performs an exact integration of the original spectrum,
    assuming it is a step function constant within each pixel, onto the new
    logarithmically-spaced pixels. This preserves the total flux.
    When `flux=False` (default), the integrated flux is divided by the width
    of each new pixel, preserving the flux density (e.g., in units of
    erg/(s cm^2 A)). This mode is generally recommended as it preserves the
    spectral shape.

    Parameters
    ----------
    lam : array_like
        Wavelength values. This can be either a 2-element array specifying the
        minimum and maximum wavelengths `[lam_min, lam_max]` for a regularly
        sampled spectrum, or a 1-D array with the central wavelength of each
        pixel for an irregularly sampled spectrum.
        - If `lam` has two elements, it defines the central wavelengths of the
          first and last pixels. The wavelength scale is assumed to be linear.
          This method is faster for regular sampling.
        - If `lam` is a 1-D array, it provides the central wavelength for each
          spectral pixel, allowing for arbitrary irregular sampling. The pixel
          edges are assumed to be the midpoints between adjacent wavelengths.

        Example for uniform wavelength sampling from FITS keywords::

            lam = CRVAL1 + CDELT1 * np.arange(NAXIS1)

    spec : array_like
        The input spectrum or an array of spectra to be rebinned. This can be a
        1-D array `spec[npixels]` or a 2-D array `spec[npixels, nspec]`.
    velscale : float, optional
        The desired velocity scale in km/s per pixel for the output spectrum.
        If not provided, it is computed to produce the same number of output
        pixels as the input. If specified, it determines the number of pixels
        and the wavelength scale of the output.
    oversample : int, default=1
        Oversampling factor. A value greater than 1 increases the number of
        output pixels, which can help prevent degradation of spectral
        resolution, especially over extended wavelength ranges, and avoid
        aliasing. An `oversample` of 1 results in approximately the same
        number of output pixels as input pixels.
    flux : bool, default=False
        Determines whether to preserve total flux or flux density.
        - If `True`, the total flux is conserved. The flux in each new pixel
          is proportional to its wavelength width (`dlam`), which can alter
          the visual shape of the spectrum.
        - If `False`, the flux density is conserved. The rebinned spectrum will
          closely overlap the original spectrum when plotted.

        Example of plotting the output::

            # With flux=True, the spectral shape changes
            plt.plot(np.exp(ln_lam), specNew)
            plt.plot(np.linspace(lam[0], lam[1], spec.size), spec)

            # With flux=False, the shapes are nearly identical
            plt.plot(np.exp(ln_lam), specNew)
            plt.plot(np.linspace(lam[0], lam[1], spec.size), spec)

    Returns
    -------
    spec_new : ndarray
        The logarithmically-rebinned spectrum or array of spectra.
    ln_lam : ndarray
        The natural logarithm of the wavelength for the new pixel grid. This
        represents the geometric mean of the wavelength at the borders of
        each pixel.
    velscale : float
        The velocity scale per pixel in km/s.

    """
    lam, spec = np.asarray(lam, dtype=float), np.asarray(spec, dtype=float)
    assert np.all(np.diff(lam) > 0), '`lam` must be monotonically increasing'
    n = len(spec)
    assert lam.size in [2, n], "`lam` must be either a 2-elements range or a vector with the length of `spec`"

    if lam.size == 2:
        dlam = np.diff(lam)/(n - 1)             # Assume constant dlam
        lim = lam + [-0.5, 0.5]*dlam
        borders = np.linspace(*lim, n + 1)
    else:
        lim = 1.5*lam[[0, -1]] - 0.5*lam[[1, -2]]
        borders = np.hstack([lim[0], (lam[1:] + lam[:-1])/2, lim[1]])
        dlam = np.diff(borders)

    ln_lim = np.log(lim)
    c = 299792.458                          # Speed of light in km/s

    if velscale is None:
        m = int(n*oversample)               # Number of output elements
        velscale = c*np.diff(ln_lim)/m      # Only for output (eq. 8 of Cappellari 2017, MNRAS)
        velscale = velscale.item()          # Make velscale a scalar
    else:
        ln_scale = velscale/c
        m = int(np.diff(ln_lim)/ln_scale)   # Number of output pixels

    newBorders = np.exp(ln_lim[0] + velscale/c*np.arange(m + 1))

    if lam.size == 2:
        k = ((newBorders - lim[0])/dlam).clip(0, n-1).astype(int)
    else:
        k = (np.searchsorted(borders, newBorders) - 1).clip(0, n-1)

    specNew = np.add.reduceat((spec.T*dlam).T, k)[:-1]    # Do analytic integral of step function
    specNew.T[...] *= np.diff(k) > 0                      # fix for design flaw of reduceat()
    specNew.T[...] += np.diff(((newBorders - borders[k]))*spec[k].T)    # Add to 1st dimension

    if not flux:
        specNew.T[...] /= np.diff(newBorders)   # Divide 1st dimension

    # Output np.log(wavelength): natural log of geometric mean
    ln_lam = 0.5*np.log(newBorders[1:]*newBorders[:-1])

    return specNew, ln_lam, velscale

###############################################################################
#
# NAME:
#   DETERMINE_GOODPIXELS
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Leiden, 9 September 2005
#   V1.0.1: Made a separate routine and included additional common emission lines.
#       MC, Oxford 12 January 2012
#   V2.0.0: Translated from IDL into Python. MC, Oxford, 10 December 2013
#   V2.0.1: Updated line list. MC, Oxford, 8 January 2014
#   V2.0.2: Use redshift instead of velocity as input for higher accuracy at large z.
#       MC, Lexington, 31 March 2015
#   V2.0.3: Includes `width` keyword after suggestion by George Privon (Univ. Florida).
#       MC, Oxford, 2 July 2018
#   V2.0.4: More exact determination of limits. MC, Oxford, 28 March 2022


def determine_goodpixels(ln_lam, lam_range_temp, redshift=0, width=800):
    """Generates a list of good pixel indices for masking emission lines.

    This function identifies the indices of pixels to be included in a pPXF
    fit, effectively masking regions contaminated by a standard set of gas
    emission lines. It also masks pixels at the edges of the template's
    wavelength range.

    This is a wrapper for ``determine_mask()`` and is the recommended way to
    generate the ``goodpixels`` input for pPXF.

    Parameters
    ----------
    ln_lam : array_like
        Natural logarithm of the wavelength for each pixel of the log-rebinned
        galaxy spectrum (``np.log(wave)``), in Angstroms.
    lam_range_temp : array_like
        A 2-element array ``[lam_min, lam_max]`` specifying the rest-frame
        wavelength range of the stellar templates, in Angstroms.
    redshift : float, default: 0
        An estimate of the galaxy's redshift.
    width : float, default: 800
        The full width in km/s of the velocity window to mask around each
        emission line.

    Returns
    -------
    ndarray
        An array of integer indices corresponding to the pixels to be
        included in the pPXF fit.

    """
    return np.flatnonzero(determine_mask(ln_lam, lam_range_temp, redshift, width))

###############################################################################


def determine_mask(ln_lam, lam_range_temp, redshift=0, width=800):
    """Generates a boolean mask to exclude regions of emission lines.

    This function creates a boolean mask to identify pixels suitable for
    fitting with pPXF. It masks out regions potentially contaminated by a
    pre-defined set of common gas emission lines and also excludes pixels
    at the spectral edges, based on the template wavelength range.

    In the returned mask, a value of ``True`` indicates a "good" pixel that
    should be included in the fit, while ``False`` indicates a pixel to be
    excluded.

    Parameters
    ----------
    ln_lam : array_like
        Natural logarithm of the wavelength for each pixel of the log-rebinned
        galaxy spectrum (``np.log(wave)``), in Angstroms.
    lam_range_temp : array_like
        A 2-element array ``[lam_min, lam_max]`` specifying the rest-frame
        wavelength range of the stellar templates, in Angstroms.
    redshift : float, default: 0
        An estimate of the galaxy's redshift.
    width : float, default: 800
        The full width in km/s of the velocity window to mask around each
        emission line.

    Returns
    -------
    numpy.ndarray
        A boolean array with the same shape as `ln_lam`. `True` indicates
        a pixel to be included in the pPXF fit.

    """
#                     -----[OII]-----    Hdelta   Hgamma   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha   -----[SII]-----
    lines = np.array([3726.03, 3728.82, 4101.76, 4340.47, 4861.33, 4958.92, 5006.84, 6300.30, 6548.03, 6583.41, 6562.80, 6716.47, 6730.85])
    # width/2 of masked gas emission region in km/s
    dv = np.full_like(lines, width)
    c = 299792.458  # speed of light in km/s

    flag = False
    for line, dvj in zip(lines, dv):
        flag |= (ln_lam > np.log(line*(1 + redshift)) - dvj/c) \
            & (ln_lam < np.log(line*(1 + redshift)) + dvj/c)

    # Mask edges of stellar library
    flag |= ln_lam > np.log(lam_range_temp[1]*(1 + redshift)) - 900/c
    flag |= ln_lam < np.log(lam_range_temp[0]*(1 + redshift)) + 900/c

    return ~flag

###############################################################################


def _wave_convert(lam):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param lam - Wavelength in Angstroms
    :return: conversion factor

    """
    lam = np.asarray(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return fact

###############################################################################


def vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms

    """
    return lam_vac/_wave_convert(lam_vac)

###############################################################################


def air_to_vac(lam_air):
    """
    Convert air to vacuum wavelengths

    :param lam_air - Wavelength in Angstroms
    :return: lam_vac - Wavelength in Angstroms

    """
    return lam_air*_wave_convert(lam_air)

###############################################################################
# NAME:
#   GAUSSIAN
#
# MODIFICATION HISTORY:
#   V1.0.0: Written using analytic pixel integration.
#       Michele Cappellari, Oxford, 10 August 2016
#   V2.0.0: Define lines in frequency domain for a rigorous
#       convolution within pPXF at any sigma, including sigma=0.
#       Introduced `pixel` keyword for optional pixel convolution.
#       MC, Oxford, 26 May 2017
#   V2.0.1: Removed Scipy next_fast_len usage. MC, Oxford, 25 January 2019


def gaussian(ln_lam_temp, line_wave, FWHM_gal, pixel=True):
    """Creates a Gaussian Line Spread Function (LSF).

    This function generates an instrumental Gaussian Line Spread Function (LSF),
    which can be optionally integrated analytically within pixels. When used as
    a template in pPXF, it is rigorously insensitive to undersampling.

    Parameters
    ----------
    ln_lam_temp : array_like
        Natural logarithm of the template wavelengths (``np.log(wavelength)``)
        in Angstroms.
    line_wave : array_like
        Wavelengths of the emission lines in Angstroms.
    FWHM_gal : float or callable
        Instrumental FWHM in Angstroms. This can be a scalar or a function
        that returns the FWHM for a given wavelength.
        - If a function is provided, the sigma returned by pPXF will be the
          intrinsic dispersion, corrected for instrumental effects.
        - To measure the *observed* dispersion (ignoring instrumental effects),
          set `FWHM_gal=0`. The LSFs then become Dirac delta functions.
    pixel : bool, default: True
        If True, performs analytic integration of the LSF over the pixels.

    Returns
    -------
    ndarray
        An array of shape `(ln_lam_temp.size, line_wave.size)` containing the
        LSF for each line.

    Notes
    -----
    The function implements equations (14) and (15) of `Westfall et al. (2019)
    <https://ui.adsabs.harvard.edu/abs/2019AJ....158..231W>`_.

    The returned LSF is normalized such that `LSF.sum(axis=0) == 1`.

    To handle potential undersampling rigorously, the Gaussian is defined
    analytically in the frequency domain and transformed numerically to the
    time domain. This ensures that the convolution within pPXF is exact to
    machine precision for any sigma, including sigma=0.

    When the LSF is not severely undersampled and `pixel=False`, the output
    is nearly indistinguishable from a standard normalized Gaussian:

    .. code-block:: python

        dx = np.diff(ln_lam_temp).mean()
        xsig = FWHM_gal / (2 * np.sqrt(2 * np.log(2))) / line_wave / dx
        x = (ln_lam_temp[:, None] - np.log(line_wave)) / dx
        gauss = np.exp(-0.5 * (x / xsig)**2)
        gauss /= (np.sqrt(2 * np.pi) * xsig)

    """
    line_wave = np.asarray(line_wave)

    if callable(FWHM_gal):
        FWHM_gal = FWHM_gal(line_wave)

    n = ln_lam_temp.size
    npad = 2**int(np.ceil(np.log2(n)))
    nl = npad//2 + 1  # Expected length of rfft

    dx = (ln_lam_temp[-1] - ln_lam_temp[0])/(n - 1)   # Delta\ln\lam
    x0 = (np.log(line_wave) - ln_lam_temp[0])/dx      # line center
    w = np.linspace(0, np.pi, nl)[:, None]            # Angular frequency

    # Gaussian with sigma=xsig and center=x0,
    # optionally convolved with a unitary pixel UnitBox[]
    # analytically defined in frequency domain
    # and numerically transformed to time domain
    xsig = FWHM_gal/2.355/line_wave/dx    # sigma in pixels units
    rfft = np.exp(-0.5*(w*xsig)**2 - 1j*w*x0)
    if pixel:
        rfft *= np.sinc(w/(2*np.pi))

    line = np.fft.irfft(rfft, n=npad, axis=0)

    return line[:n, :]

###############################################################################
# NAME:
#   EMISSION_LINES
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 7 January 2014
#   V1.1.0: Fixes [OIII] and [NII] doublets to the theoretical flux ratio.
#       Returns line names together with emission lines templates.
#       MC, Oxford, 3 August 2014
#   V1.1.1: Only returns lines included within the estimated fitted wavelength range.
#       This avoids identically zero gas templates being included in the pPXF fit
#       which can cause numerical instabilities in the solution of the system.
#       MC, Oxford, 3 September 2014
#   V1.2.0: Perform integration over the pixels of the Gaussian line spread function
#       using the new function gaussian(). Thanks to Eric Emsellem for the suggestion.
#       MC, Oxford, 10 August 2016
#   V1.2.1: Allow FWHM_gal to be a function of wavelength. MC, Oxford, 16 August 2016
#   V1.2.2: Introduced `pixel` keyword for optional pixel convolution.
#       MC, Oxford, 3 August 2017
#   V1.3.0: New `tie_balmer` keyword to assume intrinsic Balmer decrement.
#       New `limit_doublets` keyword to limit ratios of [OII] & [SII] doublets.
#       New `vacuum` keyword to return wavelengths in vacuum.
#       MC, Oxford, 31 October 2017
#   V1.3.1: Account for the varying pixel size in Angstrom, when specifying the
#       weights for the Balmer series with tie_balmer=True. Many thanks to
#       Kyle Westfall (Santa Cruz) for reporting this bug. MC, Oxford, 10 April 2018
#   V1.3.2: Include more Balmer lines when fitting with tie_balmer=False.
#       MC, Oxford, 8 April 2022
#   V1.4.0: Include extra emission lines from Table 1 of Belfiore+19.
#       MC, Oxford, 28 June 2022


def emission_lines(ln_lam_temp, lam_range_gal, FWHM_gal, pixel=True,
                   tie_balmer=False, limit_doublets=False, vacuum=False):
    """Generates Gaussian emission line templates for pPXF.

    This function creates an array of Gaussian templates for gas emission
    lines, intended for use with pPXF.

    .. note::
        This routine is a template. Users are welcome to copy, modify, and
        distribute it to accommodate their specific needs for different
        emission lines.

    The templates typically represent the instrumental Line Spread Function
    (LSF) at the wavelength of each emission line. When using these templates,
    pPXF fits for the intrinsic (astrophysical) velocity dispersion of the gas.
    Alternatively, if `FWHM_gal=0` is provided, the lines are treated as
    delta functions, and pPXF returns the observed dispersion, which is a
    combination of the intrinsic and instrumental broadening.

    The function includes options to handle common physical constraints:
    - The [OI], [OIII], and [NII] doublets are fixed at theoretical flux ratios.
    - The [OII] and [SII] doublets can be constrained to physically plausible
      flux ratios via the `limit_doublets` parameter.
    - The Balmer series can be fixed to a theoretical decrement via the
      `tie_balmer` parameter.

    Parameters
    ----------
    ln_lam_temp : array_like
        Natural logarithm of the template wavelengths in Angstroms. This
        should match the wavelength grid of the stellar templates.
    lam_range_gal : array_like
        A 2-element array specifying the estimated rest-frame wavelength
        range of the galaxy spectrum. Typically calculated as::

            lam_range_gal = np.array([np.min(wave), np.max(wave)]) / (1 + z)

    FWHM_gal : float, callable, or dict
        Instrumental resolution (FWHM) in Angstroms. This can be:
        - A scalar value for constant FWHM.
        - A function `f(wavelength)` that returns the FWHM for given wavelengths.
        - A dictionary `{'lam': lam, 'fwhm': fwhm}` specifying the FWHM at
          each pixel of the galaxy spectrum.
    pixel : bool, default: True
        If True, analytically integrates the Gaussian LSF over each pixel for
        higher accuracy.
    tie_balmer : bool, default: False
        If True, ties the Balmer lines to a theoretical decrement (Case B
        recombination, T=1e4 K, n=100 cm^-3).

        .. important::
            This option assumes the input spectrum has flux units
            proportional to `erg/(cm**2 s A)`.

    limit_doublets : bool, default: False
        If True, constrains the [OII] and [SII] doublet ratios to physically
        allowed ranges. This is done by modeling each doublet as a linear
        combination of two templates representing the minimum and maximum
        allowed ratios. An alternative is to use the `constr_templ` keyword
        in pPXF.

        .. important::
            When using this keyword, the two output fluxes (`flux_1`
            and `flux_2`) for a doublet do not represent the actual fluxes of the
            two lines, but the weights of the two doublet templates. If the two
            templates have line ratios `rat_1` and `rat_2`, the actual fitted
            ratio and total flux are::

                flux_total = flux_1 + flux_2
                ratio_fit = (rat_1*flux_1 + rat_2*flux_2)/flux_total

            EXAMPLE: For the [SII] doublet, the adopted ratios for the templates are::

                ratio_d1 = flux([SII]6716/6731) = 0.44
                ratio_d2 = flux([SII]6716/6731) = 1.43.

            When pPXF prints (and returns in pp.gas_flux)::

                flux([SII]6731_d1) = flux_1
                flux([SII]6731_d2) = flux_2

            the total flux and true lines ratio of the [SII] doublet are::

                flux_total = flux_1 + flux_2
                ratio_fit([SII]6716/6731) = (0.44*flux_1 + 1.43*flux_2)/flux_total

            Similarly, for [OII], the adopted ratios for the templates are::

                ratio_d1 = flux([OII]3729/3726) = 0.28
                ratio_d2 = flux([OII]3729/3726) = 1.47.

            When pPXF prints (and returns in pp.gas_flux)::

                flux([OII]3726_d1) = flux_1
                flux([OII]3726_d2) = flux_2

            the total flux and true lines ratio of the [OII] doublet are::

                flux_total = flux_1 + flux_2
                ratio_fit([OII]3729/3726) = (0.28*flux_1 + 1.47*flux_2)/flux_total

    vacuum : bool, default: False
        If True, assumes the provided line wavelengths are in vacuum. By
        default, they are assumed to be in air.

    Returns
    -------
    emission_lines : numpy.ndarray
        An array of shape `(ln_lam_temp.size, n_lines)` containing the
        gas emission line templates.
    line_names : numpy.ndarray
        An array of strings with the name for each gas template.
    line_wave : numpy.ndarray
        An array of the central wavelength for each gas template.

    """

    if isinstance(FWHM_gal, dict):
        FWHM_gal1 = lambda lam: np.interp(lam, FWHM_gal["lam"], FWHM_gal["fwhm"])
    else:
        FWHM_gal1 = FWHM_gal

    #        Balmer:     H10       H9         H8        Heps    Hdelta    Hgamma    Hbeta     Halpha
    balmer = np.array([3798.983, 3836.479, 3890.158, 3971.202, 4102.899, 4341.691, 4862.691, 6564.632])  # vacuum wavelengths

    if tie_balmer:

        # Balmer decrement for Case B recombination (T=1e4 K, ne=100 cm^-3)
        # from Storey & Hummer (1995) https://ui.adsabs.harvard.edu/abs/1995MNRAS.272...41S
        # In electronic form https://cdsarc.u-strasbg.fr/viz-bin/Cat?VI/64
        # See Table B.7 of Dopita & Sutherland (2003) https://www.amazon.com/dp/3540433627
        # Also see Table 4.2 of Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/
        wave = balmer
        if not vacuum:
            wave = vac_to_air(wave)
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        ratios = np.array([0.0530, 0.0731, 0.105, 0.159, 0.259, 0.468, 1, 2.86])
        # Account for varying log-sampled pixel size in Angstrom
        ratios *= wave[-2]/wave
        emission_lines = gauss @ ratios
        line_names = ['Balmer']
        w = (lam_range_gal[0] < wave) & (wave < lam_range_gal[1])
        line_wave = np.mean(wave[w]) if np.any(w) else np.mean(wave)

    else:

        line_wave = balmer
        if not vacuum:
            line_wave = vac_to_air(line_wave)
        line_names = ['H10', 'H9', 'H8', 'Heps', 'Hdelta', 'Hgamma', 'Hbeta', 'Halpha']
        emission_lines = gaussian(ln_lam_temp, line_wave, FWHM_gal1, pixel)

    if limit_doublets:

        # The line ratio of this doublet lam3727/lam3729 is constrained by
        # atomic physics to lie in the range 0.28--1.47 (e.g. fig.5.8 of
        # Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #       -----[OII]-----
        wave = [3727.092, 3729.875]    # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[OII]3726_d1', '[OII]3726_d2']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        doublets = gauss @ [[1, 1], [0.28, 1.47]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

        # The line ratio of this doublet lam6717/lam6731 is constrained by
        # atomic physics to lie in the range 0.44--1.43 (e.g. fig.5.8 of
        # Osterbrock & Ferland (2006) https://www.amazon.co.uk/dp/1891389343/).
        # We model this doublet as a linear combination of two doublets with the
        # maximum and minimum ratios, to limit the ratio to the desired range.
        #        -----[SII]-----
        wave = [6718.294, 6732.674]    # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[SII]6731_d1', '[SII]6731_d2']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        doublets = gauss @ [[0.44, 1.43], [1, 1]]  # produces *two* doublets
        emission_lines = np.column_stack([emission_lines, doublets])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    else:

        # Here the two doublets are free to have any ratio
        #         -----[OII]-----     -----[SII]-----
        wave = [3727.092, 3729.875, 6718.294, 6732.674]  # vacuum wavelengths
        if not vacuum:
            wave = vac_to_air(wave)
        names = ['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731']
        gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
        emission_lines = np.column_stack([emission_lines, gauss])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, wave)

    # Here the lines are free to have any ratio
    #       -----[NeIII]-----    HeII      HeI
    wave = [3968.59, 3869.86, 4687.015, 5877.243]  # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    names = ['[NeIII]3968', '[NeIII]3869', 'HeII4687', 'HeI5876']
    gauss = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel)
    emission_lines = np.column_stack([emission_lines, gauss])
    line_names = np.append(line_names, names)
    line_wave = np.append(line_wave, wave)

    ######### Doublets with fixed ratios #########

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OIII]-----
    wave = [4960.295, 5008.240]    # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])
    # single template for this doublet
    line_names = np.append(line_names, '[OIII]5007_d')
    line_wave = np.append(line_wave, wave[1])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #        -----[OI]-----
    wave = [6302.040, 6365.535]    # vacuum wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [1, 0.33]
    emission_lines = np.column_stack([emission_lines, doublet])
    # single template for this doublet
    line_names = np.append(line_names, '[OI]6300_d')
    line_wave = np.append(line_wave, wave[0])

    # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
    #       -----[NII]-----
    wave = [6549.860, 6585.271]    # air wavelengths
    if not vacuum:
        wave = vac_to_air(wave)
    doublet = gaussian(ln_lam_temp, wave, FWHM_gal1, pixel) @ [0.33, 1]
    emission_lines = np.column_stack([emission_lines, doublet])

    # single template for this doublet
    line_names = np.append(line_names, '[NII]6583_d')
    line_wave = np.append(line_wave, wave[1])

    # Only include lines falling within the estimated fitted wavelength range.
    #
    w = (lam_range_gal[0] < line_wave) & (line_wave < lam_range_gal[1])
    emission_lines = emission_lines[:, w]
    line_names = line_names[w]
    line_wave = line_wave[w]

    print('Emission lines included in gas templates:')
    print(line_names)

    return emission_lines, line_names, line_wave

###############################################################################
# NAME:
#   GAUSSIAN_FILTER1D
#
# MODIFICATION HISTORY:
#   V1.0.0: Written as a replacement for the Scipy routine with the same name,
#       to be used with variable sigma per pixel. MC, Oxford, 10 October 2015
#   V1.1.0: Introduced `mode` keyword. MC, Oxford, 22 April 2022
#   V1.2.0: Removed `mode` keyword and changed edge treatment to agree with
#       ndimage.gaussian_filter1d(mode='constant'). MC, Oxford, 28 June 2024

def gaussian_filter1d(spec, sig):
    """
    **************************************************************************

    **This function is deprecated: one should use `ppxf_util.varsmooth`
    instead, which is superior in every situation.**

    **************************************************************************

    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    `scipy.ndimage.gaussian_filter1d(mode='constant')
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter1d.html>`_
    When creating a template library for SDSS data (4000 pixels long),
    this implementation is 60x faster than a naive for-loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig
    """
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(1 + 4*np.max(sig))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= gau.sum(0)  # Normalize kernel

    n = spec.size
    a = np.zeros((m, n + 2*p))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, j: j + n] = spec

    conv_spectrum = (a[:, p: -p]*gau).sum(0)

    return conv_spectrum

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 25 November 2016
#   V1.0.1: Set `edgecolors` keyword in pcolormesh.
#       MC, Oxford, 14 March 2017
#   V1.1.0: Allow for 1D xgrid, ygrid. Set pcolormesh(..., lw=0.3)
#       MC, Oxford, 27 August 2022

def plot_weights_2d(xgrid, ygrid, weights, xlabel="lg Age (yr)",
                    ylabel="[M/H]", title="Weights Fraction", nodots=False,
                    colorbar=True, colorbar_label="", **kwargs):
    """
    Plot an image of the 2-dim weights, as a function of xgrid and ygrid.
    This function allows for non-uniform spacing in x or y.

    """
    if (xgrid.ndim == ygrid.ndim == 1) and (xgrid.size, ygrid.size) == weights.shape:
        ygrid, xgrid = np.meshgrid(ygrid, xgrid)  # note swapped x -- y

    assert weights.ndim == 2, "`weights` must be 2-dim"
    assert (xgrid.shape == ygrid.shape == weights.shape), \
        'Input arrays (xgrid, ygrid, weights) must have the same shape'

    x = xgrid[:, 0]  # Grid centers
    y = ygrid[0, :]
    xb = (x[1:] + x[:-1])/2  # internal grid borders
    yb = (y[1:] + y[:-1])/2
    xb = np.hstack([1.5*x[0] - x[1]/2, xb, 1.5*x[-1] - x[-2]/2])   # 1st/last border
    yb = np.hstack([1.5*y[0] - y[1]/2, yb, 1.5*y[-1] - y[-2]/2])

    # pcolormesh() is used below to allow for irregular spacing in the
    # sampling of the stellar population parameters (e.g. metallicity)

    ax = plt.gca()
    pc = plt.pcolormesh(xb, yb, weights.T, edgecolors='face', lw=0.3, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if not nodots:
        plt.plot(xgrid, ygrid, 'w,')
    if colorbar:
        cb = plt.colorbar(pc)
        cb.set_label(colorbar_label)
        plt.sca(ax)  # Activate main plot before returning

    return pc

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 8 February 2018
#   V1.0.1: Changed imports for pPXF as a package. MC, Oxford, 16 April 2018
#   V1.0.2: Removed Scipy next_fast_len usage. MC, Oxford, 25 January 2019


def convolve_gauss_hermite(templates, velscale, start, npix,
                           velscale_ratio=1, sigma_diff=0, vsyst=0):
    """
    Convolve a spectrum, or a set of spectra, arranged into columns of an array,
    with a LOSVD parametrized by the Gauss-Hermite series.

    This is intended to reproduce what pPXF does for the convolution and it
    uses the analytic Fourier Transform of the LOSVD introduced in
    Cappellari (2017) http://adsabs.harvard.edu/abs/2017MNRAS.466..798C

    EXAMPLE::

        ...
        pp = ppxf(templates, galaxy, noise, velscale, start,
                  degree=4, mdegree=4, velscale_ratio=ratio, vsyst=dv)

        spec = convolve_gauss_hermite(templates, velscale, pp.sol, galaxy.size,
                                      velscale_ratio=ratio, vsyst=dv)

        The spectrum below is equal to pp.bestfit to machine precision

        spectrum = (spec @ pp.weights)*pp.mpoly + pp.apoly

    :param templates: array[npix_temp, ntemp] (or vector[npix_temp]) of log rebinned spectra
    :param velscale: velocity scale c*Delta(ln_lam) in km/s
    :param start: parameters of the LOSVD [vel, sig, h3, h4,...] in km/s
    :param npix: number of desired output pixels (must be npix <= npix_temp)
    :return: array[npix_temp, ntemp] (or vector[npix_temp]) with the convolved templates

    """
    npix_temp = templates.shape[0]
    templates = templates.reshape(npix_temp, -1)
    start = np.array(start)  # make copy
    start[:2] /= velscale    # convert velocities to pixels
    vsyst /= velscale

    npad = 2**int(np.ceil(np.log2(npix_temp)))
    templates_rfft = np.fft.rfft(templates, npad, axis=0)
    lvd_rfft = losvd_rfft(start, 1, start.shape, templates_rfft.shape[0],
                          1, vsyst, velscale_ratio, sigma_diff)

    conv_temp = np.fft.irfft(templates_rfft*lvd_rfft[:, 0], npad, axis=0)
    conv_temp = rebin(conv_temp[:npix*velscale_ratio, :], velscale_ratio)

    return conv_temp.squeeze()

###############################################################################
# MODIFICATION HISTORY:
#   V1.0.0: Written. Michele Cappellari, Oxford, 15 March 2022
#   V1.1.0: Moved from, sps_util to ppxf_util and request spectra as input.
#       MC, Oxford, 4 April 2022

class synthetic_photometry:
    """
    Purpose
    -------

    Returns the flux in the given band for a single spectrum or a set of
    spectra arranged as columns of an array.

    The fluxes are in units of `ergs/(s cm^2 A)` and are consistent with the
    input spectra for pPXF. The function also returns the effective and pivot
    wavelengths of each band, for each spectrum.

    The filter wavelength is shifted by the input redshift, but the flux
    density is not adjusted for cosmological effects. The user is expected to
    explicitly account for the factor `(1 + z)` when needed.

    Parameters
    ----------

    lam_spec : array_like of shape (n_pixels,)
        Restframe wavelength in Angstrom for every pixel of `spectra`. This can
        be non-uniformly sampled.
    spectra : array_like of shape (n_pixels, n_spectra) or (n_pixels, ...)
        Single spectrum vector or multiple spectra arranged as columns of an
        array. Fluxes must be proportional to `ergs/(s cm^2 A)` units.
    redshift : float
        Approximate redshift of the galaxy under study.
    bands : string array_like of shape (n_bands,)
        String identifying the filter. The matching is done using partial
        string matching with the file FILTER.RES.txt.
        For example, if the filter in the file is "Johnson B-band", one can
        use "B-band" as long as it does not match other filters in the file.
        Additional filters can be added to the file under
        "ppxf/examples/FILTER.RES.txt".

    Returns
    -------

    Attributes of the ``synthetic_photometry`` class.

    .flux : array_like of shape (n_bands, n_spectra,) or (n_bands, ...)
        Fluxes for all spectra in all bands, normalized as in eq.(15) of
        `Cappellari (2023) <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_.
    .lam_eff : array_like of shape (n_bands, n_spectra,) or (n_bands, ...)
        Effective wavelength for each band, weighted by each spectrum,
        as in eq.(22) of `Cappellari (2023)`_.
    .lam_piv : array_like of shape (n_bands)
        Source-independent pivot wavelength for each band, as in eq.(17) of 
        `Cappellari (2023)`_. This is used to convert exactly 
        ``<f(lam)> = <f(nu)>c/lam_piv^2``, using the observed 
        ``lam_piv*(1 + z)`` pivot wavelength.
    .ok : array_like of shape (n_bands)
        Boolean array indicating whether the input `bands` are within the
        wavelength range of the spectra at the input redshift.
        Fluxes and wavelengths with corresponding `ok` values of False are
        returned as np.nan and should not be used.
    """

    def __init__(self, lam_spec, spectra, bands,
                 redshift=0, filters_file=None, quiet=False):

        assert len(lam_spec) == len(spectra), \
            "`lam_spec`  must have the same number of elements as `spectra.shape[0]`"

        if filters_file is None:
            ppxf_dir = resources.files('ppxf')  # path of ppxf package
            filters_file = ppxf_dir / 'examples/FILTER.RES.txt'

        bands = np.atleast_1d(bands)
        lam_eff, flux = np.empty((2, len(bands), *spectra.shape[1:]))
        lam_piv = np.empty(len(bands))
        ok = np.empty(len(bands), dtype=bool)
        for j, band in enumerate(bands):
            lam_eff[j], lam_piv[j], flux[j], ok[j] = synthetic_photometry_one_band(
                lam_spec, spectra, band, redshift, filters_file)
            if not quiet:
                if ok[j]:
                    print(f"{j + 1:3d}: {band}")
                else:
                    print(f"{j + 1:3d} --- Outside template: {band}")

        self.lam_piv = lam_piv
        self.lam_eff = lam_eff
        self.flux = flux
        self.ok = ok

###############################################################################


def synthetic_photometry_one_band(lam_spec, spectra, band, redshift, filters_file):
    """
    Compute the average fluxes inside the `band` for a single spectrum or a set
    of `spectra` arranged as columns of an array, all with common vector of
    wavelengths `lam_spec`.

    `flux` uses eq.(15), `lam_piv` eq.(17) and `lam_eff` eq.(22)
    of Cappellari (2023, MNRAS)

    """
    lam_resp, response = read_filter(band, filters_file)
    lam_resp /= 1 + redshift    
    lam_in_fwhm = lam_resp[response > 0.5*np.max(response)]     # I want FWHM fully covered
    ok = np.all((lam_in_fwhm >= lam_spec[0]) & (lam_in_fwhm <= lam_spec[-1]))

    if ok:
        fil = np.interp(lam_spec, lam_resp, response, left=0, right=0)
        fdlam = fil*np.gradient(lam_spec)
        filam2 = fdlam*lam_spec
        filam3 = filam2*lam_spec
        int1 = filam2.sum()                 # Integrate[S[lam]*lam, lam]
        int2 = filam3.sum()                 # Integrate[S[lam]*lam^2, lam]
        int3 = (spectra.T @ filam2).T       # Integrate[g[lam]*S[lam]*lam, lam]        
        int4 = (spectra.T @ filam3).T       # Integrate[g[lam]*S[lam]*lam^2, lam]
        int5 = (fdlam/lam_spec).sum()       # Integrate[S[lam]/lam, lam]
        flux = int3/int1
        with np.errstate(invalid='ignore'):
            lam_eff = np.where(int3 > 0, int4/int3, int2/int1)
        lam_piv = np.sqrt(int1/int5)
    else:
        lam_eff = flux = np.full(spectra.shape[1:], np.nan)
        lam_piv = np.nan

    return lam_eff, lam_piv, flux, ok

###############################################################################


def read_filter(band, filters_file):
    """
    Reads a filter response function from a text file containing multiple filters.

    The file should contain filters, each starting with a header line that includes
    the filter name and an integer specifying the number of rows with response values.
    Each subsequent row contains the wavelength (in Angstroms) and response value in
    the second and third columns, respectively. The absolute normalization of the
    response function is irrelevant.

    Parameters
    ----------
    band : str
        Name or substring of the filter to match in the file.
    filters_file : str or Path
        Path to the filter response file.

    Returns
    -------
    lam : ndarray
        Wavelengths in Angstroms.
    resp : ndarray
        Filter response values.

    Examples
    --------
    File format example::

        4 Johnson B-band ...
        1   3000   0
        2   3500   0.5
        3   4000   0.3
        4   4500   0
        3 Johnson V-band ...
        1   4500   0
        2   5000   0.5
        3   5500   0

    """
    with open(filters_file) as f:
        for ln in f:
            if band in ln:
                nrows = int(ln.split()[0])
                lines = [next(f) for _ in range(nrows)]
                break
        else:
            raise ValueError(f"Filter '{band}' not found in file '{filters_file}'")
    
    _, lam, resp = np.array([ln.split() for ln in lines], float).T

    return lam, resp

###############################################################################


def mag_sun(bands, redshift=0, system='AB', quiet=True):
    """
    Computes the absolute magnitude of the Sun in the specified photometric bands.

    This function calculates the absolute magnitude of the Sun in either the AB or Vega
    magnitude system for the given set of bands. The calculation can optionally be
    shifted to a specified redshift by multiplying the wavelength by ``(1 + redshift)``.
    No cosmological dimming is applied to the spectrum flux.

    Parameters
    ----------
    bands : str or array_like of str
        Name(s) of the photometric bands for which to compute the Sun's magnitude.
    redshift : float, optional
        Redshift to apply to the wavelengths before computing the flux. Default is 0.
    system : {'AB', 'Vega'}, optional
        Photometric system to use for the calculation. Default is 'AB'.
    quiet : bool, optional
        If False, prints information about the bands being processed. Default is True.

    Returns
    -------
    sun_mag : ndarray
        Absolute magnitude of the Sun in each of the specified bands.

    Examples
    --------
    >>> mag_sun(['SDSS/u', 'SDSS/g', 'SDSS/r'])
    array([6.39, 5.12, 4.68])

    """
    ppxf_dir = resources.files('ppxf')        # path of ppxf package
    filename = ppxf_dir / 'sps_models/spectra_sun_vega.npz'
    a = np.load(filename)                       # Spectra in erg/s/cm2/A at 10pc
    flux_sun, lam = a["flux_sun"], a["lam"]     # lam in Angstrom

    if system == 'AB':
        c_as = 299792458e+10                # speed of light in A/s
        flux_ref = 3631e-23*c_as/lam**2     # AB flux in erg/s/cm2/A
    elif system == 'Vega':
        flux_ref = a["flux_vega"]

    flux = np.column_stack([flux_sun, flux_ref])
    p = synthetic_photometry(lam, flux, bands, redshift=redshift, quiet=quiet)
    sun_mag = -2.5*np.log10(np.divide(*p.flux.T))

    return sun_mag

###############################################################################


def mag_spectrum(lam, spectrum, bands='SDSS/r', redshift=0, system='AB', quiet=True):
    """
    Computes the apparent magnitude from a spectrum in the specified photometric bands.

    This function calculates the apparent magnitude in either the AB or Vega
    magnitude system for the given set of bands, using a spectrum in units of
    erg/s/cm^2/A. The calculation can optionally be shifted to a specified
    redshift by multiplying the wavelength by ``(1 + redshift)``. No cosmological
    dimming is applied to the spectrum flux.

    Parameters
    ----------
    lam : array_like, shape (n_pixels,)
        Wavelengths in Angstroms for each pixel in the spectrum.
    spectrum : array_like, shape (n_pixels,)
        Spectrum flux density in erg/s/cm^2/A.
    bands : str or array_like of str, optional
        Name(s) of the photometric bands for which to compute the magnitude.
        Default is 'SDSS/r'.
    redshift : float, optional
        Redshift to apply to the wavelengths before computing the flux.
        Default is 0.
    system : {'AB', 'Vega'}, optional
        Photometric system to use for the calculation. Default is 'AB'.
    quiet : bool, optional
        If False, prints information about the bands being processed. Default is True.

    Returns
    -------
    mag : ndarray
        Apparent magnitude in each of the specified bands. If redshift is nonzero,
        the spectrum is redshifted before computing the magnitude, but no
        cosmological dimming is applied.

    Examples
    --------
    >>> mag_spectrum(lam, spectrum, bands=['SDSS/u', 'SDSS/g', 'SDSS/r'])
    array([17.23, 16.10, 15.85])

    """
    p2 = synthetic_photometry(lam, spectrum, bands, redshift=redshift, quiet=quiet)

    if system == 'AB':
        c_as = 299792458e+10                # speed of light in A/s
        flux = 3631e-23*c_as/lam**2         # AB flux in erg/s/cm2/A
    elif system == 'Vega':
        ppxf_dir = resources.files('ppxf')  # path of ppxf package
        filename = ppxf_dir / 'sps_models/spectra_sun_vega.npz'
        a = np.load(filename)               # Spectra in erg/s/cm2/A at 10pc
        flux, lam = a["flux_vega"], a["lam"]

    p1 = synthetic_photometry(lam, flux, bands, redshift=redshift, quiet=quiet)
    mag = -2.5*np.log10(p2.flux/p1.flux)

    return mag

##############################################################################


def interp(xout, xin, yin):
    """Applies `numpy.interp` to the last dimension of `yin`"""

    yout = [np.interp(xout, xin, y) for y in yin.reshape(-1, xin.size)]

    return np.reshape(yout, (*yin.shape[:-1], -1))

##############################################################################
# NAME:
#   varsmooth
#
# MODIFICATION HISTORY:
#   V1.0.0: Michele Cappellari, Oxford, 3 September 2022
#   V1.1.0: Faster convolution for a scalar sigma. MC, Oxford, 12 November 2023
#   V1.1.1: Removed dependency on legacy scipy.interpolate.interp1d using
#       faster loop over np.interp. MC, Oxford, 26 April 2024

def varsmooth(x, y, sig_x, xout=None, oversample=1):
    """
    Convolves a signal with a Gaussian kernel of spatially varying width.

    This function performs an accurate convolution of a 1D signal (`y`) with a
    Gaussian kernel whose standard deviation (`sig_x`) can vary along the
    coordinate `x`. The convolution is efficiently computed using the Fast
    Fourier Transform (FFT) and the analytical Fourier transform of a
    Gaussian, similar to the approach in the pPXF method. This ensures
    accuracy even when the Gaussian kernel is significantly undersampled.

    This method is generally preferred over standard convolution techniques,
    even for a constant Gaussian width, due to its superior handling of
    undersampling.

    The implementation is based on Algorithm 1 in `Cappellari (2023)
    <https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C>`_

    Parameters
    ----------
    x : array_like
        Coordinates corresponding to the input signal `y`.
    y : array_like
        Input 1D signal or an array where each column is a signal.
    sig_x : float or array_like
        Standard deviation (sigma) of the Gaussian kernel at each point `x`,
        in the same units as `x`.
        If `sig_x` is a scalar, it implies a constant sigma, and `x`
        must be uniformly sampled.
    xout : array_like, optional
        Coordinates at which to evaluate the convolved signal. If None
        (default), the output is evaluated at the input `x` coordinates (or
        their stretched equivalent if `sig_x` is variable).
    oversample : float, optional
        Oversampling factor applied internally before convolution, particularly
        useful when `sig_x` is variable. Default is 1.

    Returns
    -------
    yout : array_like
        The convolved signal, with the same shape as `y` but evaluated at
        coordinates `xout` (if provided) or an internal grid.

    """
    assert len(x) == len(y), "`x` and `y` must have the same length"

    if np.isscalar(sig_x):
        dx = np.diff(x)
        assert np.all(np.isclose(dx[0], dx)), "`x` must be uniformly spaced, when `sig_x` is a scalar"
        n = len(x)
        sig_max = sig_x*(n - 1)/(x[-1] - x[0])
        y_new = y.T
    else:
        assert len(x) == len(sig_x), "`x` and `sig_x` must have the same length"
        
        # Stretches spectrum to have equal sigma in the new coordinate
        sig = sig_x/np.gradient(x)
        sig = sig.clip(0.1)   # Clip to >=0.1 pixels
        sig_max = np.max(sig)*oversample
        xs = np.cumsum(sig_max/sig)
        n = int(np.ceil(xs[-1] - xs[0]))
        x_new = np.linspace(xs[0], xs[-1], n)
        y_new = interp(x_new, xs, y.T)

    # Convolve spectrum with a Gaussian using analytic FT like pPXF
    npad = 2**int(np.ceil(np.log2(n)))
    ft = np.fft.rfft(y_new, npad)
    w = np.linspace(0, np.pi*sig_max, ft.shape[-1])
    ft_gau = np.exp(-0.5*w**2)
    yout = np.fft.irfft(ft*ft_gau, npad)[..., :n]

    if not np.isscalar(sig_x):
        if xout is not None:
            xs = np.interp(xout, x, xs)  # xs is 1-dim
        yout = interp(xs, x_new, yout)

    return yout.T
