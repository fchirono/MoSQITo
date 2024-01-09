# -*- coding: utf-8 -*-
"""
Test Fluctuation Strength implementation


Author:
    Fabio Casagrande Hirono
    Jan 2024
"""

# Optional package import
try:
    import pytest
except ImportError:
    raise RuntimeError(
        "In order to perform the tests you need the 'pytest' package."
        )

import numpy as np

# Local application imports
from mosqito.sq_metrics import fluctuation_strength
from mosqito.sq_metrics.fluctuation_strength.utils import (
    _create_am_sin, _create_am_bbn, _create_fm_sin)


@pytest.mark.fluctuation_strength  # to skip or run only fluctuation strength tests
def test_fluctuation_strength_AM_sin():
    """
    Test function for the Fluctuation Strength calculation: a 60 dB SPL, 1 kHz
    tone, 100% amplitude-modulated at 4 Hz modulation frequency produces 1
    vacil [1]
    
    References
    ----------
    [1] H Fastl, E Zwicker, "Psychoacoustics: Facts and Models" (3rd Ed),
    Springer, 2007.
    """
    
    # -------------------------------------------------------------------------
    # create test signal
    fs = 48000
    dt = 1/fs
    
    T = 1
    t = np.linspace(0, T-dt, int(T*fs))
    
    # carrier frequency
    fc = 1000
    
    # modulation signal: unit-amplitude sine wave, 4 Hz
    fm = 4
    xm = 1*np.cos(2*np.pi*fm*t)
    
    spl_level = 60         # dB SPL (RMS)
    
    am_signal = _create_am_sin(spl_level, fc, xm, fs)
    
    # -------------------------------------------------------------------------
    
    f_vacil_am = fluctuation_strength(am_signal, fs)
    
    assert f_vacil_am == 1


if __name__ == "__main__":
    test_fluctuation_strength_AM_sin()
