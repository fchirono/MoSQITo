# -*- coding: utf-8 -*-
"""
Test Fluctuation Strength implementation


Author:
    Fabio Casagrande Hirono
    Jan 2024
    


References:
    
    [1] H Fastl, E Zwicker, "Psychoacoustics: Facts and Models" (3rd Ed),
    Springer, 2007.

    [2] R. Sottek et al, "Perception of Fluctuating Sounds", DAGA 2021
    URL: https://pub.dega-akustik.de/DAGA_2021/data/articles/000087.pdf
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
    vacil [1].
    
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
    
    # carrier level [in dB SPL], frequency
    spl_level = 60
    fc = 1000
    
    # modulation signal: unit-amplitude sine wave, 4 Hz
    fm = 4
    xm = 1*np.cos(2*np.pi*fm*t)
    
    am_signal = _create_am_sin(spl_level, fc, xm, fs)
    
    # -------------------------------------------------------------------------

    f_vacil_am = fluctuation_strength(am_signal, fs)
    
    assert f_vacil_am == 1



def fluct_strength_AMtone_fm(fm_vector):
    """ This equation (Eq. 1 from Sottek et al [2]) models the effect
    of the modulation frequency 'fm' on the Fluctuation Strength of a 70 dB SPL,
    1 kHz tone at 40 dB (100%) modulation depth. It shows a bandpass behavior,
    with a peak around 'f_AM' = 7.22 Hz, and is normalised to the value of 
    Fluctuation Strength for the same tone at 'fm' = 8 Hz modulation frequency.
    
    References
    ----------
    [2] R. Sottek et al, "Perception of Fluctuating Sounds", DAGA 2021
    https://pub.dega-akustik.de/DAGA_2021/data/articles/000087.pdf
    """
    
    f_AM = 7.22
    
    a_AM = np.array([0.31 if (f<f_AM) else 0.82 for f in fm_vector])
    b_AM = np.array([0.71 if (f<f_AM) else 1.13 for f in fm_vector])
    
    # FS_AM(fm) / FS_AM(8 Hz)
    FS_AM_fm = (1.05
             / (np.abs(1 + a_AM*np.abs((fm_vector/f_AM)
                                       - (f_AM/fm_vector))**2 )**b_AM))
    
    return FS_AM_fm


# %% Recreate Figures from Sottek et al - DAGA 2021 (Ref. [2])

import matplotlib.pyplot as plt

# Figure 1
fm = np.logspace(-2, 5, 64, base=2)
FS = fluct_strength_AMtone_fm(fm)

plt.figure()
plt.semilogx(fm, FS)
plt.grid()
plt.xticks(ticks=np.logspace(-2, 5, 8, base=2),
           labels=[f'{x:.02f}' for x in np.logspace(-2, 5, 8, base=2)])
plt.xlim([0.25, 32])
plt.xlabel(r'$f_m [Hz]$', fontsize=15)
plt.ylabel(r'$F_{AM}(f_m) \ / \ F_{AM}(8 Hz)$', fontsize=15)
plt.title('Fig. 1 from Sottek et al (DAGA 2021)', fontsize=15)


# Figure 2

# %%

if __name__ == "__main__":
    test_fluctuation_strength_AM_sin()
