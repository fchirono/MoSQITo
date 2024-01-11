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
plt.close('all')
from mosqito.utils.conversion import freq2bark, bark2freq


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
        Array of Fluctuation Strength values, in vacil.
        
    
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
    
    FS_AM = FS_AM_8Hz*np.array([FS_AM_norm_8Hz(f) for f in fm])
    
    return FS_AM


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
        Array of Fluctuation Strength values, in vacil.
        
    
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
    
    FS_FM = FS_FM_4Hz*np.array([FS_FM_norm_4Hz(f) for f in fm]) 
    
    return FS_FM


def fluct_strength_AM_FMtone_L(L, modulation='AM'):
    """ This equation (Eq. 3 from Sottek et al [2]) models the effect
    of the sound level 'L' (in dB SPL) on the Fluctuation Strength of a
    reference tone of 70 dB SPL, either amplitude-modulated (AM) or
    frequency-modulated (FM), as per the parameters for Eqs. 1 and 2 from the
    same Reference.
    
    Parameters
    ----------
    L : numpy.array
        Sound levels, in dB SPL.
        
    modulation : {'AM', 'FM'}
        Type of modulation, default is 'AM'.
    
    Returns
    -------
    FS_L_norm70dB : numpy.array
        Array of Fluctuation Strength values, normalised to the value for
        reference tone at 70 dB SPL.
        
    
    References
    ----------
    [2] R. Sottek et al, "Perception of Fluctuating Sounds", DAGA 2021
    https://pub.dega-akustik.de/DAGA_2021/data/articles/000087.pdf
    """
    
    check_modulation = (isinstance(modulation, str)
                        and modulation.upper() in ['AM', 'FM'])
    assert check_modulation, "'modulation' must be a string containing either 'AM' or 'FM'!"
    
    
    if isinstance(L, (int, float)):
        L = np.array([L])
    
    if modulation=='AM':
        a_L = 0.121
        b_L = 3.243
    
    elif modulation=='FM':
        a_L = 0.384
        b_L = 1.702
    
    
    # FS(L) / FS(70 dB)
    FS_L_norm70dB = a_L * b_L**(L/40)
    
    return FS_L_norm70dB



def fluct_strength_FMtone_deltaF_fc(fc, delta_f):
    """ This equation (Eq. 4 from Sottek et al [2]) models the effect
    of the frequency deviation 'delta_f' (in Hz) or of the carrier frequency
    'fc' on the Fluctuation Strength of a frequency-modulated (FM) reference
    tone of 70 dB SPL.

    Parameters
    ----------
    delta_f : (Nd,)-shaped numpy.array
        Frequency deviation value(s), in Hz.
    
    fc : (Nc,)-shaped numpy.array
        Carrier frequency value(s), in Hz.
    
    
    Returns
    -------
    FS_DeltaF_norm : (Nd, Nc)-shaped numpy.array
        Array of Fluctuation Strength values, normalised to the value for a
        reference FM tone
        
    
    Notes
    -----
    For an equation describing the effects of the carrier frequency 'fc' on
    a FM tone of fixed
    
    References
    ----------
    [2] R. Sottek et al, "Perception of Fluctuating Sounds", DAGA 2021
    https://pub.dega-akustik.de/DAGA_2021/data/articles/000087.pdf
    """
    
    if isinstance(fc, (int, float)):
        fc = np.array([fc])
    
    if isinstance(delta_f, (int, float)):
        delta_f = np.array([delta_f])
    
    fc = fc[:, np.newaxis]
    delta_f = delta_f[np.newaxis, :]
    
    delta_z = freq2bark(fc + delta_f) - freq2bark(fc - delta_f)
    
    # FS(delta z) / FS_ref
    FS_freq_dev_norm = 0.65*delta_z / np.sqrt(1 + (0.35*delta_z)**2)
    
    return np.squeeze(delta_z), np.squeeze(FS_freq_dev_norm)


# %% -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Figure 3
L_AM = np.linspace(50, 90, 50)
FS_AM_L = fluct_strength_AM_FMtone_L(L_AM)
    
L_FM = np.linspace(40, 80, 50)
FS_FM_L = fluct_strength_AM_FMtone_L(L_FM, modulation='FM')

plt.figure()
plt.subplot(121)
plt.plot(L_AM, FS_AM_L)
plt.grid()
plt.xticks(ticks=np.linspace(50, 90, 5),
            labels=[f'{x}' for x in np.linspace(50, 90, 5)])
plt.xlim([50, 90])
plt.ylim([0, 3.5])
plt.xlabel(r'$L [dB]$', fontsize=15)
plt.ylabel(r'$F_{AM}(L) \ / \ F_{AM}(70 dB)$', fontsize=15)

plt.subplot(122)
plt.plot(L_FM, FS_FM_L)
plt.grid()
plt.xticks(ticks=np.linspace(40, 80, 3),
            labels=[f'{x}' for x in np.linspace(40, 80, 3)])
plt.xlim([40, 80])
plt.ylim([0, 3.5])
plt.xlabel(r'$L [dB]$', fontsize=15)
plt.ylabel(r'$F_{FM}(L) \ / \ F_{FM}(70 dB)$', fontsize=15)

plt.suptitle('Fig. 3 from Sottek et al (DAGA 2021)', fontsize=15)

plt.tight_layout()


# -----------------------------------------------------------------------------
# Figure 5

# Freq range approximately covers 'delta_z' range from 0 to 6 Bark
delta_f = np.linspace(0, 675, 101, endpoint=True)

delta_z1, FS_FM_deltaF = fluct_strength_FMtone_deltaF_fc(fc=1500, delta_f=delta_f)

plt.figure()
plt.plot(delta_z1, FS_FM_deltaF)
plt.grid()
plt.ylim([0, 2])
plt.xlabel(r'Freq deviation $\Delta z$ [Bark$_{HMS}$] (for fixed $f_c$=1500 Hz)',
           fontsize=13)
plt.ylabel(r'$F_{BW}$($\Delta$z) / $F_{BW, ref}$', fontsize=15)
plt.title(r'Fig. 5 from Sottek et al, DAGA 2021',
          fontsize=14)
plt.text(1., 0.25, r'*$F_{BW, ref} = F_{BW}( f_c$=1500 Hz, $\Delta f$ = 200 Hz)',
         fontsize=14)


# %%

if __name__ == "__main__":
    test_fluctuation_strength_AM_sin()
