# -*- coding: utf-8 -*-
"""
Validation for Fluctuation Strength implementation

This code assesses the Fluctuation Strength of synthetic, stationary
amplitude-modulated (AM) or frequency-modulated (FM) sounds, and compares them
to perceived values obtained via jury tests and approximated by analytical
expressions by Sottek et al [2].

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
        "In order to perform the tests you need the 'pytest' package.")


import numpy as np

import matplotlib.pyplot as plt
plt.close('all')

# Local application imports
from mosqito.sq_metrics import fluctuation_strength
from mosqito.sq_metrics.fluctuation_strength.utils import (
    _create_am_sin, _create_fm_sin)


# %% Global parameters

fs = 48000
dt = 1/fs

T = 1.5
t = np.linspace(0, T-dt, int(T*fs))

save_fig = False


# %% Equations from Sottek et al (DAGA 2021)

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
        
    """
    
    if isinstance(fc, (int, float)):
        fc = np.array([fc])
    
    if isinstance(delta_f, (int, float)):
        delta_f = np.array([delta_f])
    
    fc = fc[:, np.newaxis]
    delta_f = delta_f[np.newaxis, :]
    
    # Eq. 6.1 from Fastl and Zwicker - Psychoacoustics, 3rd Ed
    myfreq2bark = lambda f: 13*np.arctan(0.76*f/1000) + 3.5*np.arctan(f/7500)**2
    
    delta_z = myfreq2bark(fc + delta_f) - myfreq2bark(fc - delta_f)
    
    # FS(delta z) / FS_ref
    FS_freq_dev_norm = 0.65*delta_z / np.sqrt(1 + (0.35*delta_z)**2)
    
    return np.squeeze(delta_z), np.squeeze(FS_freq_dev_norm)


# %% Figure 1 from Sottek et al (DAGA 2021)


# Test parameters for AM tone:
Lp = 70     # Level, dB SPL
fc = 1000   # carrier frequency, Hz

# 40 dB (100%) modulation strength - modulating signal has unitary amplitude


# varying modulation frequency
N_test1 = 64
fm = np.logspace(-2, 5, N_test1, base=2)

FS_AM = np.zeros(N_test1)

# evaluate FS for various AM tones
for i, f in enumerate(fm):
    xm = np.sin(2*np.pi*f*t)
    x = _create_am_sin(Lp, fc, xm, fs)
    FS_AM[i] = fluctuation_strength(x, fs)

# normalise to FS of reference tone (8 Hz modulation rate)
xm_ref = np.sin(2*np.pi*8*t)
x_ref = _create_am_sin(Lp, fc, xm_ref, fs)
FS_AM *= 1/fluctuation_strength(x_ref, fs)

# analytical expression from Sottek et al (DAGA 2021)
FS_AM_fm = fluct_strength_AMtone_fm(fm)/fluct_strength_AMtone_fm(8)

# test for 20% tolerance
test = (FS_AM < 1.2*FS_AM_fm).all() and (FS_AM > 0.8*FS_AM_fm).all()

        
plt.figure(1, figsize=(8, 6))
plt.semilogx(fm, FS_AM_fm, label='Eq. 1 (Sottek et al, DAGA 2021)')
plt.semilogx(fm, 0.8*FS_AM_fm, 'C0--', label='20% tolerance')
plt.semilogx(fm, 1.2*FS_AM_fm, 'C0--')

plt.semilogx(fm, FS_AM, 'C1*:', label='MoSQITo implementation')

plt.grid()
plt.xticks(ticks = np.logspace(-2, 5, 8, base=2),
            labels = [f'{x:.02f}' for x in np.logspace(-2, 5, 8, base=2)])
plt.xlim([0.25, 32])
plt.ylim([0, 1.4])
plt.legend()
plt.xlabel(r'$f_m$ [Hz]', fontsize=15)
plt.ylabel(r'$F_{AM}$($f_m$) / $F_{AM}$(8 Hz)', fontsize=15)
plt.title(r'FS as function of $f_m$ (AM tone, 1 kHz, 100% mod. depth)',
          fontsize=14)

if test:
    plt.text( 0.5, 0.5, "Test passed (20% tolerance not exceeded)", fontsize=13,
        horizontalalignment="center", verticalalignment="center",
        transform=plt.gca().transAxes, bbox=dict(facecolor="green", alpha=0.3))
else:
    plt.text(0.5, 0.5, "Test not passed", fontsize=13,
             horizontalalignment="center", verticalalignment="center",
             transform=plt.gca().transAxes, bbox=dict(facecolor="red", alpha=0.3))

plt.tight_layout()

if save_fig:
    plt.savefig('Fig1_Sottek_etal_DAGA2021.png')


# %% Figure 2 from Sottek et al (DAGA 2021)


# Test parameters for FM tone:
Lp2 = 70     # Level, dB SPL
fc2 = 1500   # carrier frequency, Hz

delta_f2 = 700   # freq deviation, Hz

# varying modulation frequency
N_test2 = 64
fm2 = np.logspace(-2, 5, N_test2, base=2)

FS_FM = np.zeros(N_test2)

# evaluate FS for various FM tones
for i, f in enumerate(fm2):
    xm2 = np.sin(2*np.pi*f*t)
    x2 = _create_fm_sin(Lp2, fc2, xm2, delta_f2, fs)
    FS_FM[i] = fluctuation_strength(x2, fs)

# normalise to FS of reference tone (4 Hz modulation rate)
xm_ref2 = np.sin(2*np.pi*4*t)
x_ref2 = _create_fm_sin(Lp2, fc2, xm_ref2, delta_f2, fs)
FS_AM *= 1/fluctuation_strength(x_ref2, fs)

# analytical expression from Sottek et al (DAGA 2021)
FS_FM_fm = fluct_strength_FMtone_fm(fm)/fluct_strength_FMtone_fm(4)

# test for 20% tolerance
test2 = (FS_FM < 1.2*FS_FM_fm).all() and (FS_FM > 0.8*FS_FM_fm).all()


plt.figure(2, figsize=(8, 6))
plt.semilogx(fm, FS_FM_fm, label='Eq. 2 (Sottek et al, DAGA 2021)')
plt.semilogx(fm, 0.8*FS_FM_fm, 'C0--', label='20% tolerance')
plt.semilogx(fm, 1.2*FS_FM_fm, 'C0--')

plt.semilogx(fm, FS_FM, 'C1*:', label='MoSQITo implementation')

plt.grid()
plt.xticks(ticks=np.logspace(-2, 5, 8, base=2),
            labels=[f'{x:.02f}' for x in np.logspace(-2, 5, 8, base=2)])
plt.xlim([0.25, 32])
plt.ylim([0, 1.2])
plt.xlabel(r'$f_m$ [Hz]', fontsize=15)
plt.ylabel(r'$F_{FM}$($f_m$) / $F_{FM}$(4 Hz)', fontsize=15)
plt.title(r'FS as function of $f_m$ (FM tone, 1.5 kHz, 700 Hz freq deviation)',
          fontsize=14)


if test2:
    plt.text( 0.5, 0.5, "Test passed (20% tolerance not exceeded)", fontsize=13,
        horizontalalignment="center", verticalalignment="center",
        transform=plt.gca().transAxes, bbox=dict(facecolor="green", alpha=0.3))
else:
    plt.text(0.5, 0.5, "Test not passed", fontsize=13,
             horizontalalignment="center", verticalalignment="center",
             transform=plt.gca().transAxes, bbox=dict(facecolor="red", alpha=0.3))

plt.tight_layout()

if save_fig:
    plt.savefig('Fig2_Sottek_etal_DAGA2021.png')


# %% Figure 3 from Sottek et al (DAGA 2021)

# ----------------------------------------------------------------------------
# Part (a): AM tones

# Test parameters for AM tone:
fc3 = 1000   # carrier frequency, Hz
fm3 = 4      # modulation frequency [Hz]

# 40 dB (100%) modulation strength - modulating signal has unitary amplitude

N_test3 = 64
FS_AM3 = np.zeros(N_test3)

# range of levels, in dB SPL
L_AM = np.linspace(50, 90, N_test3)

# evaluate FS for various AM tones
for i, l in enumerate(L_AM):
    xm3 = np.sin(2*np.pi*fm3*t)
    x3 = _create_am_sin(l, fc3, xm3, fs)
    FS_AM3[i] = fluctuation_strength(x3, fs)

# normalise to FS of reference tone (70 dB)
xm_ref3 = np.sin(2*np.pi*4*t)
x_ref3 = _create_am_sin(70, fc3, xm_ref3, fs)
FS_AM3 *= 1/fluctuation_strength(x_ref3, fs)

FS_AM_L = fluct_strength_AM_FMtone_L(L_AM)

# test for 20% tolerance
test3 = (FS_AM3 < 1.2*FS_AM_L).all() and (FS_AM3 > 0.8*FS_AM_L).all()

    
plt.figure('3a', figsize=(8, 6))

plt.plot(L_AM, FS_AM_L, label='Eq. 3 (Sottek et al (DAGA 2021)')
plt.plot(L_AM, 0.8*FS_AM_L, 'C0--', label='20% tolerance')
plt.plot(L_AM, 1.2*FS_AM_L, 'C0--')

plt.semilogx(L_AM, FS_FM, 'C1*:', label='MoSQITo implementation')

plt.grid()
plt.xticks(ticks=np.linspace(50, 90, 5),
            labels=[f'{x}' for x in np.linspace(50, 90, 5)])
plt.xlim([50, 90])
plt.ylim([0, 3.5])
plt.xlabel(r'$L$ [dB]', fontsize=15)
plt.ylabel(r'$F_{AM}$($L$) / $F_{AM}$(70 dB)', fontsize=15)

plt.title(r'FS as function of $L$ (AM tone, 1 kHz, 100% mod. depth)',
          fontsize=14)

if test3:
    plt.text( 0.5, 0.5, "Test passed (20% tolerance not exceeded)", fontsize=13,
        horizontalalignment="center", verticalalignment="center",
        transform=plt.gca().transAxes, bbox=dict(facecolor="green", alpha=0.3))
else:
    plt.text(0.5, 0.5, "Test not passed", fontsize=13,
             horizontalalignment="center", verticalalignment="center",
             transform=plt.gca().transAxes, bbox=dict(facecolor="red", alpha=0.3))

plt.tight_layout()


# ----------------------------------------------------------------------------
# Part (b): FM tone


# Test parameters for FM tone:
fc3b = 1500         # carrier frequency, Hz
fm3b = 4            # modulation frequency [Hz]
delta_f3b = 700     # frequency deviation

N_test3b = 64
FS_FM3b = np.zeros(N_test3b)

# range of levels, in dB SPL
L_FM3 = np.linspace(40, 80, N_test3)

# evaluate FS for various AM tones
for i, l in enumerate(L_FM3):
    xm3b = np.sin(2*np.pi*fm3b*t)
    x3b = _create_fm_sin(l, fc3b, xm3b, delta_f3b, fs)
    FS_FM3b[i] = fluctuation_strength(x3b, fs)

# normalise to FS of reference tone (70 dB)
xm_ref3b = np.sin(2*np.pi*4*t)
x_ref3b = _create_fm_sin(70, fc3b, xm_ref3b, delta_f3b, fs)
FS_FM3b *= 1/fluctuation_strength(x_ref3b, fs)

FS_FM_L = fluct_strength_AM_FMtone_L(L_FM3, modulation='FM')


# test for 20% tolerance
test3b = (FS_FM3b < 1.2*FS_FM_L).all() and (FS_FM3b > 0.8*FS_FM_L).all()

plt.figure('3b', figsize=(8, 6))
plt.plot(L_FM3, FS_FM_L, label='Eq. 3 (Sottek et al, DAGA 2021')
plt.plot(L_FM3, 0.8*FS_FM_L, 'C0--', label='20% tolerance')
plt.plot(L_FM3, 1.2*FS_FM_L, 'C0--')

plt.plot(L_FM3, FS_FM3b, 'C1*:', label='MoSQITo implementation')

plt.grid()
plt.xticks(ticks=np.linspace(40, 80, 3),
            labels=[f'{x}' for x in np.linspace(40, 80, 3)])
plt.xlim([40, 80])
plt.ylim([0, 3.5])
plt.xlabel(r'$L$ [dB]', fontsize=15)
plt.ylabel(r'$F_{FM}$($L$) / $F_{FM}$(70 dB)', fontsize=15)

plt.title(r'FS as function of $L$ (FM tone, 1.5 kHz, 700 Hz freq. deviation)',
          fontsize=14)

if test3b:
    plt.text( 0.5, 0.5, "Test passed (20% tolerance not exceeded)", fontsize=13,
        horizontalalignment="center", verticalalignment="center",
        transform=plt.gca().transAxes, bbox=dict(facecolor="green", alpha=0.3))
else:
    plt.text(0.5, 0.5, "Test not passed", fontsize=13,
             horizontalalignment="center", verticalalignment="center",
             transform=plt.gca().transAxes, bbox=dict(facecolor="red", alpha=0.3))

plt.tight_layout()


if save_fig:
    plt.savefig('Fig3_Sottek_etal_DAGA2021.png')
    
    
# %% Figure 5 from Sottek et al (DAGA 2021)
# --> modulation frequency 'fm' is unclear from paper! Assuming 'fm' = 4 Hz

# Test parameters for FM tone:
L_FM5 = 70          # tone level, dB SPL
fc5 = 1500          # carrier frequency, Hz
fm5 = 4             # modulation frequency [Hz]


# Freq. range approximately covers 'delta_z' range from 0 to 6 Bark
N_test5 = 151
delta_f5 = np.linspace(0, 675, N_test5, endpoint=True)

FS_FM5 = np.zeros(N_test5)

# evaluate FS for various FM tones
for i, d in enumerate(delta_f5):
    xm5 = np.sin(2*np.pi*fm5*t)
    x5 = _create_fm_sin(l, fc5, xm5, d, fs)
    FS_FM5[i] = fluctuation_strength(x5, fs)

# normalise to FS of reference tone (70 dB)
xm_ref5 = np.sin(2*np.pi*4*t)
x_ref5 = _create_fm_sin(L_FM5, fc3b, xm_ref5, 200, fs)
FS_FM5 *= 1/fluctuation_strength(x_ref5, fs)


delta_z1, FS_FM_deltaF = fluct_strength_FMtone_deltaF_fc(fc=fc5, delta_f=delta_f5)

# test for 20% tolerance
test5 = (FS_FM3b < 1.2*FS_FM_L).all() and (FS_FM3b > 0.8*FS_FM_L).all()


plt.figure(5, figsize=(8, 6))
plt.plot(delta_z1, FS_FM_deltaF, label='Eq. 4 (Sottek et al, DAGA 2021)')
plt.plot(delta_z1, 0.8*FS_FM_deltaF, 'C0--', label='20% tolerance')
plt.plot(delta_z1, 1.2*FS_FM_deltaF, 'C0--')

plt.plot(delta_z1, FS_FM5, 'C1*:', label='MoSQITo implementation')

plt.grid()
plt.ylim([0, 2])
plt.xlabel(r'Freq deviation $\Delta z$ [Bark$_{HMS}$] (for fixed $f_c$=1500 Hz)',
            fontsize=13)
plt.ylabel(r'$F_{BW}$($\Delta$z) / $F_{BW, ref}$', fontsize=15)

plt.text(1., 0.25, r'*$F_{BW, ref} = F_{BW}( f_c$=1500 Hz, $\Delta f$ = 200 Hz)',
          fontsize=14)

plt.title(r'FS as function of $\Delta f$ (FM tone, 1.5 kHz, 200 Hz freq. deviation, $f_m$=?)',
          fontsize=14)

if test5:
    plt.text( 0.5, 0.5, "Test passed (20% tolerance not exceeded)", fontsize=13,
        horizontalalignment="center", verticalalignment="center",
        transform=plt.gca().transAxes, bbox=dict(facecolor="green", alpha=0.3))
else:
    plt.text(0.5, 0.5, "Test not passed", fontsize=13,
             horizontalalignment="center", verticalalignment="center",
             transform=plt.gca().transAxes, bbox=dict(facecolor="red", alpha=0.3))

plt.tight_layout()

if save_fig:
    plt.savefig('Fig5_Sottek_etal_DAGA2021.png')
    
    
# %% Figure 6 from Sottek et al (DAGA 2021)
# --> modulation frequency 'fm' is unclear from paper! Assuming 'fm' = 4 Hz

# Test parameters for FM tone:
L_FM6 = 70          # tone level, dB SPL
delta_f6 = 200      # frequency deviation, Hz
fm6 = 4             # modulation frequency [Hz]

N_test6 = 151
fc6 = np.linspace(500, 9000, N_test6, endpoint=True)

FS_FM6 = np.zeros(N_test6)

# evaluate FS for various FM tones
for i, f in enumerate(fc6):
    xm6 = np.sin(2*np.pi*fm6*t)
    x6 = _create_fm_sin(L_FM6, f, xm6, delta_f6, fs)
    FS_FM6[i] = fluctuation_strength(x6, fs)

# normalise to FS of reference tone (1.5 kHz)
xm_ref6 = np.sin(2*np.pi*4*t)
x_ref6 = _create_fm_sin(L_FM6, 1500, xm_ref6, delta_f6, fs)
FS_FM6 *= 1/fluctuation_strength(x_ref6, fs)


delta_z6, FS_FM_fc = fluct_strength_FMtone_deltaF_fc(fc=fc6, delta_f=delta_f6)

# test for 20% tolerance
test6 = (FS_FM6 < 1.2*FS_FM_fc).all() and (FS_FM6 > 0.8*FS_FM_fc).all()


plt.figure(6, figsize=(8, 6))
plt.plot(fc6, FS_FM_fc, label='Eq. 4 (Sottek et al, DAGA 2021)')
plt.plot(fc6, 0.8*FS_FM_fc, 'C0--', label='20% tolerance')
plt.plot(fc6, 1.2*FS_FM_fc, 'C0--')

plt.plot(fc6, FS_FM6, 'C1*:', label='MoSQITo implementation')

plt.grid()
plt.xlim([500, 8000])
plt.ylim([0, 2])

plt.xticks(ticks=np.array([500, 1500, 4500, 8000]),
            labels=['0.5', '1.5', '4.5', '8'])
plt.xlabel(r'Carrier frequency $f_c$ [kHz] (for fixed $\Delta f$=200 Hz)',
            fontsize=13)
plt.ylabel(r'$F_{CF}$($f_c$) / $F_{CF, ref}$', fontsize=15)

plt.text(650, 1.75, r'*$F_{CF, ref} = F_{CF}( f_c$=1500 Hz, $\Delta f$ = 200 Hz)',
          fontsize=14)

plt.title(r'FS as function of $f_c$ (FM tone, 200 Hz freq. deviation, $f_m$=?)',
          fontsize=14)

if test6:
    plt.text( 0.5, 0.5, "Test passed (20% tolerance not exceeded)", fontsize=13,
        horizontalalignment="center", verticalalignment="center",
        transform=plt.gca().transAxes, bbox=dict(facecolor="green", alpha=0.3))
else:
    plt.text(0.5, 0.5, "Test not passed", fontsize=13,
             horizontalalignment="center", verticalalignment="center",
             transform=plt.gca().transAxes, bbox=dict(facecolor="red", alpha=0.3))

plt.tight_layout()

if save_fig:
    plt.savefig('Fig6_Sottek_etal_DAGA2021.png')
    
