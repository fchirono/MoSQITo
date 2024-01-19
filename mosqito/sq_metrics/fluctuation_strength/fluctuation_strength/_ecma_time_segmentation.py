# -*- coding: utf-8 -*-

import numpy as np


def _ecma_time_segmentation(signal_block, sb, sh, n_new):
    """Function used for the segmentation of a time signal into
    smaller parts of audio (blocks) following Formulas 18 to 20 (section 5.1.5)
    of ECMA-418-2:2022.

    Parameters
    ----------
    signal_block: list
        List of Numpy arrays containing bandpassed signals per critical band
    
    sb: int or list of int
        Block size, or list of block sizes per band
        
    sh: int or list of int
        Hop size, or list of hop sizes per band
    
    n_new : int
        Number of samples in signal after zero padding (Eq. 3)
    
    Returns
    -------
    block_array: list
        List of 53 two-dimensional arrays of size (nperseg, nseg)
        containing the segmented signal per critical band.
    
    time: numpy.array
        The time axis corresponding to the segmented
        signal, size (nseg,)
    """

    if isinstance(sb, int):
        sb = sb * np.ones(53, dtype=int)
        
    elif len(sb) != 53:
        raise ValueError("ERROR: len(sb) shall be either 1 or 53")
        
    if isinstance(sh, int):
        sh = sh * np.ones(53, dtype=int)
        
    elif len(sh) != 53:
        raise ValueError("ERROR: len(sh) shall be either 1 or 53")


    # ************************************************************************
    # Section 5.1.5 of ECMA-418-2, 2nd Ed. (2022)
    
    
    # Eq. (19)
    i_start = sb[0] - sb
    
    # Eq. (20) - number of blocks for each critical band
    L_last = (np.ceil( (n_new + sh) / sh) - 1).astype('int')
    
    block_array = []
    for z in range(53):
        
        signal = signal_block[z]
        
        signal_segmented = np.zeros((sb[z], L_last[z]))
        
        for L in np.arange(L_last[z]):
            
            # Eq. (18)
            indices = np.arange(L*sh[z] + i_start[z], L*sh[z] + i_start[z] + sb[z])
            signal_segmented[:, L] = signal[indices]

        block_array.append(signal)
        
    return block_array


# %% ************************************************************************
# Original version, from mosqito.utils

# # build time axis for sig
# time = np.linspace(0, (len(sig) - 1) / fs, num=len(sig))

# l = 0
# block_array = []
# time_array = []
# while l * noverlap <= len(sig) - nperseg:
#     block_array.append(sig[l * noverlap : nperseg + l * noverlap])
#     time_array.append(np.mean(time[l * noverlap : nperseg + l * noverlap]))
#     l += 1

# return np.array(block_array).T, np.array(time_array)
