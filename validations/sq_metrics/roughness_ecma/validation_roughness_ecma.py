# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:41:37 2020

@author: wantysal
"""


# Standard imports
import numpy as np
import matplotlib.pyplot as plt

# Local application imports
from mosqito.sq_metrics import roughness_dw, roughness_ecma
from mosqito.sq_metrics.roughness.utils import _create_am_bbn, _create_am_sin

from input.references import ref_zf


# %% Define range of fc and fm

fc_array = np.array([125, 250, 500, 1000, 2000, 4000, 8000])


# range from 10 to 400
fm1 = 10
fm2 = 400
fm_array = np.logspace(np.log10(fm1), np.log10(fm2), 15, base=10)


# %% Recreate Figure 11.2 from Fastl & Zwicker

linestyles = [':', ':', ':', '-', '--', '--', '--',]

plt.figure()

for i, fc in enumerate(fc_array):
    
    R = ref_zf(fc, fm_array)
    plt.loglog(fm_array, R, label=f'fc = {fc} Hz', linestyle=linestyles[i])

plt.grid()
plt.xlim([10, 400])
plt.ylim([0.07, 1])
plt.legend()
plt.xlabel(r'Modulation frequency $f_{m} [Hz]$')
plt.ylabel('Roughness [asper]')
plt.title('Fastl & Zwicker')

# %% create test signals

# preliminary definitions
fs = 48000
dt = 1/fs

T = 1.5
t = np.linspace(0, T-dt, int(T*fs))

spl = 60

for fc in fc_array:
    
    # get values from Fastl & Zwicker
    R_fz = ref_zf(fc, fm_array)
    
    R_dw = np.zeros(fm_array.shape)
    R_ecma = np.zeros(fm_array.shape)
    
    for i, fm in enumerate(fm_array):
        
        # modulating frequency
        xm = 1.0*np.sin(2*np.pi*fm*t)

        # test signal - amplitude-modulated sine wave
        x = _create_am_sin(spl, fc, xm, fs)

        # calculate Roughness using DW model
        R_dw_, _, _, _ = roughness_dw(x, fs=fs, overlap=0.5)
        R_dw[i] = np.mean(R_dw)
        
        # calculate Roughness using ECMA model
        R_ecma_, _, _, _ = roughness_ecma(x, fs)
        R_ecma[i] = np.percentile(R_ecma_, 90)
        
    plt.figure()
    plt.loglog(fm_array, R_fz, 'Fastl & Zwicker [interp]', '-')
    plt.loglog(fm_array, R_dw, 'Daniel & Weber', ':')
    plt.loglog(fm_array, R_fz, 'ECMA 418-2', '-.')
    plt.xlabel(r'Modulation Frequency $f_m$ [Hz]')
    plt.ylabel('Roughness [asper]')
    plt.title(r'Carrier frequency $f_c$={fc} Hz')
