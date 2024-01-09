# -*- coding: utf-8 -*-
"""
Base function for fluctuation strength calculation

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

import numpy as np

from mosqito.sq_metrics.fluctuation_strength.utils import _create_am_cosine, _create_am_noise, _create_fm_cosine


def fluctuation_strength(signal, fs):
    """
    Dummy function for calculating fluctuation strength.

    Parameters
    ----------
    signal: numpy.array
        Time signal values in 'Pa'.
    
    fs: float
        Sampling frequency, in Hz.

    Returns
    -------
    f_vacil: numpy.array
        Numpy array containing values of fluctuation strength of input signal,
        in vacil.
    """

    Nt = signal.shape[0]

    return np.zeros(Nt)