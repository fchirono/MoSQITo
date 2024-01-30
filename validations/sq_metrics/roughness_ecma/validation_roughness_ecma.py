# -*- coding: utf-8 -*-
"""
Validation script of Roughness ECMA 418-2, 2nd Ed (2022) standard [1]. 

Creates a series of amplitude-modulated sine waves at varying carrier
frequencies 'fc' and modulation frequencies 'fm', and compares with values
taken from Fastl & Zwicker [2], Figs. 11.1 and 11.2.

References:
    
    [1] ECMA International, "Psychoacoustic metrics for ITT equipment - Part 2
    (models based on human perception)", Standard ECMA-418-2, 2nd edition,
    Dec 2022.
    URL: https://ecma-international.org/publications-and-standards/standards/ecma-418/
    
    [2] H Fastl, E Zwicker, "Psychoacoustics: Facts and Models" (3rd Ed),
    Springer, 2007.

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""


# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from mosqito.sq_metrics import roughness_dw, roughness_ecma
from mosqito.sq_metrics.roughness.utils import _create_am_bbn, _create_am_sin

from input.references import ref_zf



# %% preliminary definitions
fs = 48000
dt = 1/fs

T = 1.5
t = np.linspace(0, T-dt, int(T*fs))

spl = 60

# %% Fig. 11.1

# number of points in plot
N_m = 15

mod_low = 0.2
mod_high = 1.0
mod_degree = np.logspace(np.log10(mod_low), np.log10(mod_high), N_m)

fc = 1000.
fm = 70


R_dw_m = np.zeros(N_m)
R_ecma_m = np.zeros(N_m)

R_FZ = mod_degree**1.6

for i, m in enumerate(mod_degree):
    
    # modulating signal
    xm = m*np.sin(2*np.pi*fm*t)

    # test signal - amplitude-modulated sine wave
    x = _create_am_sin(spl, fc, xm, fs)
    
    # calculate Roughness using DW model
    R_dw_temp, _, _, _ = roughness_dw(x, fs=fs, overlap=0.5)
    R_dw_m[i] = np.mean(R_dw_temp)
    
    # calculate Roughness using ECMA model
    R_ecma_temp, _, _, _ = roughness_ecma(x, fs)
    R_ecma_m[i] = np.percentile(R_ecma_temp, 90)
        

plt.figure()
plt.loglog(mod_degree, R_FZ, label='Fastl & Zwicker')
plt.loglog(mod_degree, R_dw_m, label='MoSQITo [Daniel & Weber]')
plt.loglog(mod_degree, R_ecma_m, label='MoSQITo [ECMA-418-2]')
plt.xlim([0.1, 1])
plt.ylim([0.07, 1])
plt.legend()
plt.xlabel(r'Degree of modulation')
plt.ylabel('Roughness [asper]')
plt.title('Fastl & Zwicker, Fig. 11.1')

# %% Fig. 11.2 - Define ranges of fc and fm

fc_all = np.array([125, 250, 500, 1000, 2000, 4000, 8000])

# lower, upper frequency in range for each fc
fm_all = np.array([[10,  10,  10,  10,  15,  15,  20],
                   [100, 150, 200, 400, 350, 300, 250]])


# Recreate Figure 11.2 from Fastl & Zwicker
N_interp_fm = 21

linestyles = [':', ':', ':', '-', '--', '--', '--',]

plt.figure()

for i, fc in enumerate(fc_all):
    
    fm = np.logspace(np.log10(fm_all[0, i]),
                     np.log10(fm_all[1, i]), N_interp_fm, base=10)

    R = ref_zf(fc, fm)
    
    plt.loglog(fm, R, label=f'fc = {fc} Hz', linestyle=linestyles[i])

plt.grid()
plt.xlim([10, 400])
plt.ylim([0.07, 1])
plt.legend()
plt.xlabel(r'Modulation frequency $f_{m} [Hz]$')
plt.ylabel('Roughness [asper]')
plt.title('Fastl & Zwicker, Fig. 11.2')

# %% create test signals

for i, fc in enumerate(fc_all):
    
    fm_array = np.logspace(np.log10(fm_all[0, i]),
                           np.log10(fm_all[1, i]), N_interp_fm, base=10)
    
    # get values from Fastl & Zwicker
    R_fz = ref_zf(fc, fm_array)
    
    R_dw = np.zeros(fm_array.shape)
    R_ecma = np.zeros(fm_array.shape)
    
    for j, fm in enumerate(fm_array):
        
        # modulating signal
        xm = 1.0*np.sin(2*np.pi*fm*t)

        # test signal - amplitude-modulated sine wave
        x = _create_am_sin(spl, fc, xm, fs)

        # plt.figure()
        # plt.plot(t, x)
        
        # calculate Roughness using DW model
        R_dw_temp, _, _, _ = roughness_dw(x, fs=fs, overlap=0.5)
        R_dw[j] = np.mean(R_dw_temp)
        
        # calculate Roughness using ECMA model
        R_ecma_temp, _, _, _ = roughness_ecma(x, fs)
        R_ecma[j] = np.percentile(R_ecma_temp, 90)
        
    plt.figure()
    plt.loglog(fm_array, R_fz, '-', linewidth=2, label='Fastl & Zwicker [interp]')
    plt.loglog(fm_array, R_dw, ':', linewidth=1.5, label='MoSQITo [Daniel & Weber]')
    plt.loglog(fm_array, R_ecma, '-.', label='MoSQITo [ECMA-418-2]')
    plt.legend()
    plt.grid()
    plt.xlim([10, 400])
    # plt.ylim([0.07, 1])
    plt.xlabel(r'Modulation Frequency $f_m$ [Hz]')
    plt.ylabel('Roughness [asper]')
    plt.title(rf'Carrier frequency $f_c$={fc} Hz')
    
