# -*- coding: utf-8 -*-
"""
Test signal for fluctuation strength calculation

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np


def create_fm_cosine(A, xm, k, fc, fs, return_aux_params=False,
                     print_info=False):
    """
    Creates a frequency-modulated (FM) signal with peak amplitude 'A', 
    cosine carrier at frequency 'fc', modulating signal 'xm', and sampling
    frequency 'fs'. The FM signal length is the same as the length of 'xm'. 
    
    Parameters
    ----------
    A: float
        Peak amplitude of the resulting FM signal.
    
    xm: numpy.array
        Numpy array containing the modulating signal.
    
    k: float
        Frequency sensitivity of the modulator. This is equal to the frequency
        deviation in Hz away from 'fc' per unit amplitude of the modulating
        signal 'xm'.
    
    fc: float
        Carrier frequency, in Hz. Must be less than 'fs/2'.
    
    fs: float
        Sampling frequency, in Hz.
    
    return_aux_params: bool, optional
        Flag declaring whether to return a dict containing auxiliary parameters.
        See notes for details. Default is False.
    
    print_info: bool, optional
        Flag declaring whether to print values for maximum frequency deviation and FM modulation index. Default is False.
    
    
    Returns
    -------
    y_fm: numpy.array
        Frequency-modulated signal with cosine carrier
    
    aux_params: dict
        Dictionary of auxiliary parameters, containing:
            'inst_freq': numpy array of instantaneous frequency of output signal;
            'max_freq_deviation': float, maximum frequency deviation from 'fc';
            'FM_modulation_index': float, FM modulation index
    """
    
     # sampling interval in seconds
    dt = 1/fs

    # instantaneous frequency of FM signal
    inst_freq = fc + k*xm
    
    # FM signal, normalised to peak amplitude 'A'
    y_fm = A*np.cos(2*np.pi* np.cumsum(inst_freq)*dt)
    
    # max frequency deviation
    f_delta = k*np.max(np.abs(xm))
    
    # FM modulation index
    m_FM = np.max(np.abs(2*np.pi*k*np.cumsum(xm)*dt))
    
    if print_info:
        print(f'\tMax freq deviation: {f_delta} Hz')
        print(f'\tFM modulation index: {m_FM:.2f} Hz')

    aux_params = {
        'inst_freq': inst_freq,
        'max_freq_deviation': f_delta,
        'FM_modulation_index': m_FM}

    if return_aux_params:
        
        return y_fm, aux_params
    
    else:
        return y_fm
