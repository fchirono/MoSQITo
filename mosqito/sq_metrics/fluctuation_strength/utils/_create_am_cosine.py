# -*- coding: utf-8 -*-
"""
Test signal for fluctuation strength calculation

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np


def create_am_cosine(A, xm, fc, fs, print_m=False):
    """
    Creates an amplitude-modulated (AM) signal with peak amplitude 'A', 
    cosine carrier at frequency 'fc', modulating signal 'xm', and sampling
    frequency 'fs'. The AM signal length is the same as the length of 'xm'. 
    
    Parameters
    ----------
    A: float
        Peak amplitude of the resulting AM signal.
    
    xm: numpy.array
        Numpy array containing the modulating signal.
    
    fc: float
        Carrier frequency, in Hz. Must be less than 'fs/2'.
    
    fs: float
        Sampling frequency, in Hz.
    
    print_m: bool, optional
        Flag declaring whether to print the calculated modulation index.
        Default is False.
    
    Returns
    -------
    y: numpy.array
        Amplitude-modulated signal with cosine carrier
        
    Notes
    -----
    The modulation index 'm' will be equal to the peak value of the modulating
    signal 'xm'. Its value can be printed by setting the optional flag
    'print_m' to True.
    """
    
    Nt = xm.shape[0]        # signal length in samples
    T = Nt/fs               # signal length in seconds
    dt = 1/fs               # sampling interval in seconds

    # vector of time samples
    t = np.linspace(0, T-dt, int(T*fs))
    
    # unitary-amplitude carrier signal
    xc = np.cos(2*np.pi*fc*t)

    # AM signal, normalised to peak amplitude 'A'
    y_am = (1 + xm)*xc/2
    y_am *= A/np.max(np.abs(y_am))

    # modulation index
    m = np.max(np.abs(xm))

    if print_m:
        print(f"AM Modulation index = {m}")
    
    if m > 1:
        print("Warning ['create_am_cosine']: modulation index m > 1\n\tSignal is overmodulated!")

    return y_am