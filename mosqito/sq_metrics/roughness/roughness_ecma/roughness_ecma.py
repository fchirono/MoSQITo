# -*- coding: utf-8 -*-
"""
Base function for Roughness calculation, according to ECMA-418-2, 2nd Ed (2022)

Author:
    Fabio Casagrande Hirono
    Jan 2024
"""


# Standard imports
import numpy as np

import matplotlib.pyplot as plt
import scipy.signal as ssig


# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma.loudness_ecma import loudness_ecma
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import _band_pass_signals
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import _ecma_time_segmentation
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import _auditory_filters_centre_freq
from mosqito.sq_metrics.loudness.loudness_ecma._preprocessing import _preprocessing

from mosqito.sq_metrics.roughness.roughness_ecma._von_hann import _von_hann
from mosqito.sq_metrics.roughness.roughness_ecma._env_noise_reduction import _env_noise_reduction
from mosqito.sq_metrics.roughness.roughness_ecma._peak_picking import _peak_picking
from mosqito.sq_metrics.roughness.roughness_ecma._weight_high_mod_rates import _weight_high_mod_rates
from mosqito.sq_metrics.roughness.roughness_ecma._est_fund_mod_rate import _est_fund_mod_rate
from mosqito.sq_metrics.roughness.roughness_ecma._weight_low_mod_rates import _weight_low_mod_rates
from mosqito.sq_metrics.roughness.roughness_ecma._interpolate_to_50Hz import _interpolate_to_50Hz
from mosqito.sq_metrics.roughness.roughness_ecma._nonlinear_transform import _nonlinear_transform
from mosqito.sq_metrics.roughness.roughness_ecma._lowpass_filtering import _lowpass_filtering


def roughness_ecma(signal, fs, sb=16384, sh=4096):
    """Calculation of the psychoacoustic roughness a signal, based on the
    scaled envelope power spectra, as described in ECMA-418-2, 2nd Ed (2022).
    
    The calculation is based on the signal Specific Loudness " N'(z, l)", 
    and the envelope of the 53 segmented, band-passed signals in each critical
    band. The signal must be sampled at 'fs'=48 kHz

    Parameters
    ----------
    signal :numpy.array  or DataTime object
        A time signal in Pa
        
    fs : float, optional
        Sampling frequency, in Hz.
        
    sb: int or list of int
        Block size, or list of block sizes per band
        
    sh: int or list of int
        Hop size, or list of hop sizes per band

    Returns
    -------
    R : (Nt,)-shaped numpy.array
        Time-dependent Roughness
        
    R_spec: (53, Nt)-shaped numpy.array
        Time-dependent Specific Rughness [asper per Bark].

    bark_axis : (53,)-shaped numpy.array
        Critical frequency scale, in Bark.
    
    time : (Nt,)-shaped numpy.array
        Time axis at 50 Hz sampling frequency.
    
    Examples
    --------
    To obtain the single-value Roughness of a signal, take the 90th percentile
    of the time-dependent Roughness 'R', discarding the first 16 samples (approx.
    300 ms):
        
    >>> R, _, _, _ = roughness_ecma(signal, fs)
    >>> R_singlevalue = np.percentile(R[16:], 90)
    """
    
    assert fs == 48000, "Sampling frequency 'fs' must be 48 kHz!"
    
    n_samples = signal.shape[0]
    
    # -------------------------------------------------------------------------
    # Preliminaries
    
    # Calculate time-dependent specific loudness N'(l, z)
    # --> N_basis is (53, L)-shaped, where L is the number of time segments
    N, N_time, N_spec, bark_axis, time_axis = loudness_ecma(signal, fs, sb, sh)
    N_basis = np.array(N_spec)
    
    # Number of time segments
    L = N_basis.shape[1]
    
    # get centre frequencies of auditory filters
    F = _auditory_filters_centre_freq()
    
    # 5.1.2 Windowing and zero-padding
    signal, n_new = _preprocessing(signal, sb, sh)

    # 5.1.3, 5.1.4 - Computaton of band-pass signals
    bandpass_signals = _band_pass_signals(signal)

    # 5.1.5 Segmentation into blocks    
    block_array, time_array = _ecma_time_segmentation(bandpass_signals,
                                                      sb, sh, n_new)

    # block_array is (53, L, sb)-shaped
    block_array = np.array(block_array)
    
    # time_array is (53, L)-shaped
    time_array = np.array(time_array)
    
    # ************************************************************************
    # 7.1.2 Envelope calculation
    
    # Envelope calculation using Hilbert Transform (Eq. 65)
    analytic_signal = ssig.hilbert(block_array)
    p_env = np.abs(analytic_signal)
    
    # .........................................................................
    # # plot envelope and bandpass signal for one segment
    
    # from mosqito.utils.conversion import bark2freq
    
    # z = 35
    # timestep_to_plot = 8
    
    # t = np.linspace(0, (sb-1)/fs, sb)
    
    # plt.figure()
    # plt.plot(t, p_env[z, timestep_to_plot, :],
    #           label='Envelope')
    # plt.plot(t, block_array[z, timestep_to_plot, :], ':',
    #           label='Bandpass Signal')
    # plt.legend()
    # plt.xlim([0, 0.03])
    # plt.xlabel('Time [s]')
    # plt.title(f'{bark_axis[z]:1.1f} Bark ({bark2freq(bark_axis[z]):1.0f} Hz)')
    # plt.tight_layout()
    
    # plt.savefig(f'02_BandpassedSignalsEnvelope.png')
    # .........................................................................
    
    # Downsampling by a total factor of 32, in two separate steps of 8 and 4
    downsampling_factor = 32
    p_env_downsampled_ = ssig.decimate(p_env, downsampling_factor//4)
    p_env_downsampled = ssig.decimate(p_env_downsampled_, 4)
    
    # new downsampled sampling freq, block size, hop size
    fs_ = fs//downsampling_factor       # 1500 Hz
    sb_ = sb//downsampling_factor       # 512 points
    sh_ = sh//downsampling_factor       # 128 points
    
    
    # .........................................................................
    # compare original and downsampled envelope
    
    # from mosqito.utils.conversion import bark2freq
    
    # z = 35
    # timestep_to_plot = 8
    
    # t = np.linspace(0, (sb-1)/fs, sb)
    # t_ = np.linspace(0, (sb_-1)/fs_, sb_)
    
    # plt.figure()
    # plt.plot(t, p_env[z, timestep_to_plot, :], 'o-',
    #           label='Envelope [original]')
    
    # # # plot every 32nd sample in p_env for visual reference
    # # plt.plot(t[::32], p_env[z, timestep_to_plot, ::32], 's-.',
    # #           color='C1', label='Envelope [every 32nd sample]')
    
    # plt.plot(t_, p_env_downsampled[z, timestep_to_plot, :], '^:',
    #           markersize=12, color='C3', label='Envelope [downsampled]')
    # plt.legend()
    # plt.grid()
    # plt.xlim([0, 0.015])
    # plt.tight_layout()
    # plt.title(f'{bark_axis[z]:1.0f} Bark ({bark2freq(bark_axis[z]):1.0f} Hz)')
    
    # plt.savefig(f'04_DownsampledSignalsEnvelope.png')
    
    # ************************************************************************
    # 7.1.3 Calculation of scaled power spectrum
    
    # get Von Hann window coefficients
    hann = _von_hann()
    
    # max Loudness per time step
    N_max = np.max(N_basis, axis=0)
    
    # Eq. 67
    phi_env_zero = np.sum( (hann*p_env_downsampled)**2 , axis=-1)
    
    # Scaled power spectrum (Eq. 66)
    non_zero = (N_max*phi_env_zero != 0)
    
    scaling = np.zeros(N_basis.shape)
    scaling[non_zero] = ( (N_basis**2)[non_zero]
                         / (N_max * phi_env_zero)[non_zero] )
    
    # 'Phi_env' is (53, L, sb_)-shaped
    Phi_env = (scaling[:, :, np.newaxis]
               * np.abs(np.sqrt(2)
                        * np.fft.fft(hann*p_env_downsampled, axis=-1) )**2 )
    
    # .........................................................................
    # plot scaled power spectrum for one time segment
    
    # timestep_to_plot = 8
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # Pspec = 10*np.log10(Phi_env[:, timestep_to_plot, :sb_//2+1])
    
    # plt.figure()
    # plt.pcolormesh(f, bark_axis, Pspec,
    #             vmax=np.max(Pspec), vmin=np.max(Pspec)-80)
    # plt.title(f'Scaled power spectrum of envelopes')
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Critical band [Bark]')
    # plt.colorbar()
    # plt.tight_layout()
    
    # # plt.savefig(f'05_ScaledEnvelopePowerSpectra.png')
    # .........................................................................
    
    # ************************************************************************
    # 7.1.4 Envelope noise reduction
    
    Phi_hat = _env_noise_reduction(Phi_env)
    
    # .........................................................................
    # plot scaled power spectrum for one time segment
    
    # timestep_to_plot = 8
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # Pspec = 10*np.log10(Phi_hat[:, timestep_to_plot, :sb_//2+1])
    
    # plt.figure()
    # plt.pcolormesh(f, bark_axis, Pspec,
    #             vmax=np.max(Pspec), vmin=np.max(Pspec)-80)
    # plt.title(f'Noise-suppressed power spectrum of envelopes')
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Critical band [Bark]')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(f'05_NoiseSuppressedPowerSpectra.png')
    
    # ************************************************************************
    # 7.1.5 Spectral weighting
    
    A = np.zeros((53, L))
    
    # for each critical freq 'z', time step 'l'...
    for z in range(53):
        for l in range(L):
            
            # # dummy variables for debugging
            # z = 30
            # l = 8
            
            # 7.1.5.1. Peak picking
            f_pi, A_pi = _peak_picking(Phi_hat[z, l], fs_)
            
            # if one or more peaks were found...
            if f_pi.size > 0:
                
                # 7.1.5.2. Weighting of high modulation rates
                A_pi_tilde = _weight_high_mod_rates(f_pi, A_pi, F[z])
                
                # 7.1.5.3. Estimation of fundamental modulation rate
                f_p_imax, f_pi_hat, A_hat = _est_fund_mod_rate(f_pi, A_pi_tilde)
                
                # 7.1.5.4. Weighting of low modulation rates
                A[z, l] = _weight_low_mod_rates(f_p_imax, f_pi_hat, A_hat, F[z])
            
                # 7.1.6. Optional entropy weighting based on randomness of the
                # modulation rate  --->>> NOT IMPLEMENTED! <<<---
                #   Requires a RPM signal with the same sampling rate as the sound
                #   pressure signal
            
            # if no peaks were found...
            else:
                A[z, l] = 0.
    
    # .........................................................................
    # plot scaled peak amplitudes
    
    # plt.figure()
    # plt.pcolormesh(time_array[0], bark_axis, A)
    # plt.title(f"Weighted peaks' magnitudes")
    # plt.xlabel('Time [s]')
    # plt.ylabel('Critical band [Bark]')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.savefig(f'08_WeightedPeaksMagnitudes.png')   
    
    
    # ************************************************************************
    # 7.1.7 Calcuation of time-dependent specific roughness
    
    # interpolate to new sampling frequency fs_50 = 50 Hz
    R_est, t_50, fs_50 = _interpolate_to_50Hz(A, time_array, n_samples, fs)
    
    # Perform nonlinear transform and calibration (Eqs. 104 to 108)
    R_hat = _nonlinear_transform(R_est)
    R_spec = _lowpass_filtering(R_hat, fs_50)
    
    # calculate time-dependent Roughness by integrating over all critical bands
    # with delta_z = 0.5
    R_l = np.sum(R_spec, axis=0) * 0.5
    
    
    return R_l, R_spec, bark_axis, t_50
