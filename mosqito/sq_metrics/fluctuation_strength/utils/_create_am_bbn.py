# -*- coding: utf-8 -*-
"""
Test signal for fluctuation strength calculation

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np


def _create_am_bbn(A, xm, fs, print_m=False):
    """
    Creates a amplitude-modulated (AM) signal with peak amplitude 'A', Gaussian
    broadband (noise) carrier, modulating signal 'xm', and sampling frequency
    'fs'. The AM signal length is the same as the length of 'xm'. 
    
    Parameters
    ----------
    A: float
        Peak amplitude of the resulting AM signal.
    
    xm: numpy.array
        Numpy array containing the modulating signal.
    
    fs: float
        Sampling frequency, in Hz.
    
    print_m: bool, optional
        Flag declaring whether to print the calculated modulation index.
        Default is False.
    

    Returns
    -------
    y: numpy.array
        Amplitude-modulated noise signal
    
        
    Notes
    -----
    The modulation index 'm' will be equal to the peak value of the modulating
    signal 'xm'. Its value can be printed by setting the optional flag
    'print_m' to True.
    
    """
    
    # signal length in samples
    Nt = xm.shape[0]
    
    # create vector of zero-mean, unitary std dev random samples
    rng = np.random.default_rng()
    xc = rng.standard_normal(Nt)

    # AM signal, normalised to peak amplitude 'A'
    y_am = (1 + xm)*xc/2
    y_am *= A/np.max(np.abs(y_am))

    # AM modulation index
    m = np.max(np.abs(xm))

    if print_m:
        print(f"AM Modulation index = {m}")
    
    if m > 1:
        print("Warning ['create_am_noise']: modulation index m > 1\n\tSignal is overmodulated!")

    return y_am