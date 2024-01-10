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


# %% Recreate Figures from Sottek et al - DAGA 2021 (Ref. [2])

import matplotlib.pyplot as plt


def fluct_strength_AMtone_fm(fm):
    """ This equation (Eq. 1 from Sottek et al [2]) models the effect
    of the modulation frequency 'fm' on the Fluctuation Strength of a reference
    tone of 70 dB SPL, 'fc'=1 kHz, amplitude-modulated (AM) at 40 dB (100%)
    modulation depth.
    
    Parameters
    ----------
    fm : numpy.array
        Modulation frequency values, in Hz.
    
    Returns
    -------
    FS_AM : numpy.array
        Array of Fluctuation Strength values for the reference tone, in vacil.
        
    
    References
    ----------
    [2] R. Sottek et al, "Perception of Fluctuating Sounds", DAGA 2021
    https://pub.dega-akustik.de/DAGA_2021/data/articles/000087.pdf
    """
    
    if isinstance(fm, (int, float)):
        fm = np.array([fm])
    
    f_AM = 7.22
      
    a_AM = lambda f: 0.31 if (f<f_AM) else 0.82
    b_AM = lambda f: 0.71 if (f<f_AM) else 1.13
      
    # FS_AM(fm) / FS_AM(8 Hz)
    FS_AM_norm_8Hz = lambda f: (
        1.05
        / (np.abs(1 + a_AM(f) * np.abs( (f/f_AM) - (f_AM/f) )**2 )**b_AM(f) ))
    
    # ------------------------------------------------------------
    # convert output units to vacil
    
    # Fluctuation Strength of 1 kHz, 70 dB SPL tone, amplitude-modulated at
    # 100% modulation and 4 Hz modulation rate is 1.3 vacil
    FS_AM_4Hz = 1.3
    
    # constant to convert function output to units of vacil
    FS_AM_8Hz = FS_AM_4Hz/FS_AM_norm_8Hz(8)
    # ------------------------------------------------------------
    
    return FS_AM_8Hz*np.array([FS_AM_norm_8Hz(f) for f in fm])


def fluct_strength_FMtone_fm(fm):
    """ This equation (Eq. 1 from Sottek et al [2]) models the effect
    of the modulation frequency 'fm' on the Fluctuation Strength of a reference
    tone of 70 dB SPL, 'fc'=1.5 kHz, frequency-modulated (FM) at 700 Hz
    frequency deviation.
    
    Parameters
    ----------
    fm : numpy.array
        Modulation frequency values, in Hz.
    
    Returns
    -------
    FS_FM : numpy.array
        Array of Fluctuation Strength values for the reference tone, in vacil.
        
    
    References
    ----------
    [2] R. Sottek et al, "Perception of Fluctuating Sounds", DAGA 2021
    https://pub.dega-akustik.de/DAGA_2021/data/articles/000087.pdf
    """
    
    if isinstance(fm, (int, float)):
        fm = np.array([fm])
    
    f_FM = 4.
      
    a_FM = lambda f: 0.46 if (f<f_FM) else 0.60
    b_FM = lambda f: 0.59 if (f<f_FM) else 1.00
      
    # FS_FM(fm) / FS_FM(4 Hz)
    FS_FM_norm_4Hz = lambda f: (
        1.0
        / (np.abs(1 + a_FM(f) * np.abs( (f/f_FM) - (f_FM/f) )**2 )**b_FM(f) ))
    
    # Fluctuation Strength of 1 kHz, 70 dB SPL tone, frequency-modulated at
    # 700 Hz frequency deviation and 4 Hz modulation rate is 2 vacil
    FS_FM_4Hz = 2.
    
    return FS_FM_4Hz*np.array([FS_FM_norm_4Hz(f) for f in fm])



# -----------------------------------------------------------------------------
# Figure 1
fm = np.logspace(-2, 5, 64, base=2)

FS_AM_fm = fluct_strength_AMtone_fm(fm)/fluct_strength_AMtone_fm(8)

plt.figure()
plt.semilogx(fm, FS_AM_fm)
plt.grid()
plt.xticks(ticks = np.logspace(-2, 5, 8, base=2),
           labels = [f'{x:.02f}' for x in np.logspace(-2, 5, 8, base=2)])
plt.xlim([0.25, 32])
plt.ylim([0, 1.2])
plt.xlabel(r'$f_m [Hz]$', fontsize=15)
plt.ylabel(r'$F_{AM}(f_m) \ / \ F_{AM}(8 Hz)$', fontsize=15)
plt.title('Fig. 1 from Sottek et al (DAGA 2021)', fontsize=15)


# Figure 2
FS_FM_fm = fluct_strength_FMtone_fm(fm)/fluct_strength_FMtone_fm(4)

plt.figure()
plt.semilogx(fm, FS_FM_fm)
plt.grid()
plt.xticks(ticks=np.logspace(-2, 5, 8, base=2),
            labels=[f'{x:.02f}' for x in np.logspace(-2, 5, 8, base=2)])
plt.xlim([0.25, 32])
plt.ylim([0, 1.2])
plt.xlabel(r'$f_m [Hz]$', fontsize=15)
plt.ylabel(r'$F_{FM}(f_m) \ / \ F_{FM}(4 Hz)$', fontsize=15)
plt.title('Fig. 2 from Sottek et al (DAGA 2021)', fontsize=15)


# %%

if __name__ == "__main__":
    test_fluctuation_strength_AM_sin()
