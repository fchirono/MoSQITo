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
def test_fluctuation_strength():
    """
    Test function for the Fluctuation Strength calculation
    """
    
    # -------------------------------------------------------------------------
    # create test signal:
    #   60 dB, 1 kHz tone, 100% amplitude-modulated at 4 Hz produces 1 vacil
    fs = 48000
    dt = 1/fs
    
    T = 1
    t = np.linspace(0, T-dt, int(T*fs))
    
    # carrier frequency
    fc = 1000
    
    # modulation signal: unit-amplitude sine wave, 4 Hz
    fm = 4
    xm = 1*np.cos(2*np.pi*fm*t)
    
    # TODO: define amplitude of carrier tone, NOT of AM signal!
    dB = 60
    A = 0.00002 * (10**(dB / 20))
    
    signal = _create_am_sin(A, xm, fc, fs)
    
    # -------------------------------------------------------------------------
    
    f_vacil = fluctuation_strength(signal, fs)
    
    assert f_vacil == 1


if __name__ == "__main__":
    test_fluctuation_strength()
