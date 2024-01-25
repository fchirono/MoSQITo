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
import scipy.ndimage as sim

# Project Imports
from mosqito.sq_metrics.loudness.loudness_ecma.loudness_ecma import loudness_ecma
from mosqito.sq_metrics.loudness.loudness_ecma._band_pass_signals import _band_pass_signals
from mosqito.sq_metrics.loudness.loudness_ecma._ecma_time_segmentation import _ecma_time_segmentation
from mosqito.sq_metrics.loudness.loudness_ecma._auditory_filters_centre_freq import _auditory_filters_centre_freq

from mosqito.sq_metrics.roughness.roughness_ecma._beta import _beta
from mosqito.sq_metrics.roughness.roughness_ecma._error_correction import _error_correction
from mosqito.sq_metrics.roughness.roughness_ecma._von_hann import _von_hann

from mosqito.utils.conversion import bark2freq


def roughness_ecma(signal, fs, sb=16384, sh=4096):
    """[*WARNING*] Roughness calculation of a signal sampled at 48 kHz,
    according to ECMA-418-2, 2nd Ed (2022).

    *************************** WARNING! ************************************
    * The code is not finished yet, and does not return correct results!    *    
    *************************************************************************

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
    R: list of numpy.array
        Time-dependent Specific Rughness [asper per Bark]. Each of the 53 element
        of the list corresponds to the time-dependent specific roughness for a
        given bark band. Can be a ragged array if a different sb/sh are used
        for each band.

    bark_axis: numpy.array
        Bark axis
    """
    
    assert fs == 48000, "Sampling frequency 'fs' must be 48 kHz!"
    
    n_samples = signal.shape[0]
    
    # ************************************************************************
    # Preliminary: calculate time-dependent specific loudness N'(l, z)
    
    # N_basis is (53, L)-shaped,, where L is the number of time segments
    N_basis, bark_axis = loudness_ecma(signal, sb, sh)
    N_basis = np.array(N_basis)
    
    L = N_basis.shape[1]
    
    F = _auditory_filters_centre_freq()
    
    # ************************************************************************
    # Section 5.1.2 of ECMA-418-2, 2nd Ed (2022)
    
    # -----------------------------------------------------------------------    
    # Apply windowing function to first 5 ms (240 samples)
    n_fadein = 240
    
    # Eq. (1)
    w_fadein = 0.5 - 0.5*np.cos(np.pi*np.arange(n_fadein)/n_fadein)
    
    signal[:240] *= w_fadein
    
    # -----------------------------------------------------------------------    
    # Calculate zero padding at start and end of signal
    sb_max = np.max(sb)
    sh_max = np.max(sh)
    
    n_zeros_start = sb_max
    
    # Eqs. (2), (3) 
    n_new = sh_max * (np.ceil((n_samples + sh_max + sb_max)/(sh_max)) - 1)
    
    n_zeros_end = int(n_new) - n_samples
    
    signal = np.concatenate( (np.zeros(n_zeros_start),
                              signal,
                              np.zeros(n_zeros_end)))
    
    # ************************************************************************
    # Sections 5.1.3 to 5.1.4 of ECMA-418-2, 2nd Ed. (2022)
    
    # Computaton of band-pass signals
    bandpass_signals = _band_pass_signals(signal)

    # ************************************************************************
    # Section 5.1.5 of ECMA-418-2, 2nd Ed. (2022)
    
    # segmentation into blocks
    block_array, time_array = _ecma_time_segmentation(bandpass_signals, sb, sh,
                                                      n_new)
    
    # block_array is (53, L, sb)-shaped, where L is number of time segments
    block_array = np.array(block_array)
    
    # time_array is (53, L)-shaped
    time_array = np.array(time_array)
    
    # ************************************************************************
    # Section 7.1.2 of ECMA-418-2, 2nd Ed. (2022)
    
    # Envelope calculation using Hilbert Transform (Eq. 65)
    analytic_signal = ssig.hilbert(block_array)
    p_env = np.abs(analytic_signal)
    
    # ------------------------------------------------------------------------
    # # plot envelope and bandpass signal for one segment
    
    # band_to_plot = 35
    # timestep_to_plot = 8
    
    # t = np.linspace(0, (sb-1)/fs, sb)
    
    # plt.figure()
    # plt.plot(t, p_env[band_to_plot, timestep_to_plot, :],
    #           label='Envelope')
    # plt.plot(t, block_array[band_to_plot, timestep_to_plot, :], ':',
    #           label='Bandpass Signal')
    # plt.legend()
    # plt.title(f'{band_to_plot} Bark ({bark2freq(band_to_plot)} Hz)')
    # ------------------------------------------------------------------------
    
    # Downsampling by a factor of 32
    downsampling_factor = 32
    p_env_downsampled_ = ssig.decimate(p_env, downsampling_factor//4)
    p_env_downsampled = ssig.decimate(p_env_downsampled_, 4)
    
    # new downsampled sampling freq, block size, hop size
    fs_ = fs//downsampling_factor       # 1500 Hz
    sb_ = sb//downsampling_factor       # 512 points
    sh_ = sh//downsampling_factor       # 128 points
    
    # ************************************************************************
    # Section 7.1.3 of ECMA-418-2, 2nd Ed. (2022)
    
    # Calculation of scaled power spectrum
    
    # get Von Hann window coefficients
    hann = _von_hann()
    
    # max Loudness (Eq. 67)
    N_max = np.max(N_basis, axis=0)
    Phi_env_zero = np.sum( (hann*p_env_downsampled)**2 , axis=-1)
    
    
    # Scaled power spectrum (Eq. 66)
    scaling = np.zeros(N_basis.shape)
    non_zero = (N_max*Phi_env_zero != 0)
    scaling[non_zero] = (N_basis**2)[non_zero] / (N_max * Phi_env_zero)[non_zero]
    
    #   'Phi_env' is (53, L, sb_)-shaped
    Phi_env = (scaling[:, :, np.newaxis]
               * np.abs(np.fft.fft(hann*p_env_downsampled))**2 )
    
    # ------------------------------------------------------------------------
    # plot scaled power spectrum for one time segment
    
    # timestep_to_plot = 8
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # plt.figure()
    # plt.pcolormesh(f, bark_axis,
    #                 10*np.log10(Phi_env[:, timestep_to_plot, :sb_//2+1]))
    # plt.title(f'Scaled power spectrum of envelopes')
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Critical band [Bark]')
    # plt.colorbar()
    # ------------------------------------------------------------------------
    
    # ************************************************************************
    # Section 7.1.4 of ECMA-418-2, 2nd Ed. (2022)
    
    # Noise reduction of the envelopes
    
    # averaging across neighboring critical bands
    two_dim_filter = np.array([1/3, 1/3, 1/3])
    
    Phi_avg = sim.convolve(Phi_env,
                           two_dim_filter[:, np.newaxis, np.newaxis],
                           mode='constant',
                           cval=0.)
    
    # ------------------------------------------------------------------------
    # plot averaged power spectrum for one time segment
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # plt.figure()
    # plt.pcolormesh(f, bark_axis,
    #                 10*np.log10(Phi_avg[:, timestep_to_plot, :sb_//2+1]))
    # plt.title(f'Averaged power spectrum of envelopes')
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Critical band [Bark]')
    # plt.colorbar()
    # ------------------------------------------------------------------------
    
    # Sum the averaged Power spectra to get an overview of all the modulation
    # patterns over time (Eq. 68)
    s_ = np.sum(Phi_avg, axis=0)
    
    # median of 's_(L, k)' between k=2 and k=255
    k_range = np.arange(2, 256)
    s_tilde = np.median(s_[:, k_range], axis=-1)

    # 's_tilde' becomes small compared to the peaks for modulated signals,
    # whereas 's_tilde' and the peaks have comparable amplitude for unmodulated
    # signals
    
    delta = 1e-10
    
    # k = 0, ..., 511
    k = np.arange(sb_)
    
    # Eq. 71
    #   --> 's_ / s_tilde' have large ratios for modulated signals
    w_tilde = (0.0856
               * (s_ / (s_tilde[:, np.newaxis] + delta))
               * np.clip(0.1891 * np.exp(0.0120 * k), 0., 1.) )
    
    # Eq. 70
    w_tilde_max = np.max(w_tilde[:, k_range], axis=-1)
    w_mask = (w_tilde >= 0.05 * w_tilde_max[:, np.newaxis])
    
    w = np.zeros(s_.shape)
    w[w_mask] = np.clip( w_tilde[w_mask] - 0.1407, 0., 1.)
    
    # 'w' tends to 0 for unmodulated signals, and to 1 for modulated signals
    # --> For white gaussian noise of 80 dB, all weights 'w' become 0 and
    #   result in a roughness of 0 Asper
    
    # ------------------------------------------------------------------------
    # plot weighting value 'w' for one time segment
    
    # df_ = fs_/sb_
    # f = np.linspace(0, fs_ - df_, sb_)[:sb_//2+1]
    
    # plt.figure()
    # plt.pcolormesh(f, time_array[0, :], w[:, :sb_//2+1])
    # plt.title(f"Weighting coefficient 'w' (Eq. 70)")
    # plt.xlabel('Freq [Hz]')
    # plt.ylabel('Time step [s]')
    # plt.colorbar()
    # ------------------------------------------------------------------------
    
    # weighted, averaged Power Spectra (Eq. 69)
    Phi_hat = Phi_avg*w
    
    # ************************************************************************
    # Section 7.1.5 of ECMA-418-2, 2nd Ed. (2022)
    
    # Spectral weighting
    
    # 7.1.5.1. Peak picking
    
    # search for local maxima in Phi_hat for k=2, ..., 255
    
    # for each critical freq...
    for z in range(53):
        
        # for each time step...
        for l in range(L):
            
            # TODO: code might not find any peaks!
            
            # find peaks, calculate their prominence (Scipy definition of
            # prominence matches the definition in ECMA-418-2, 2022)
            peaks, peaks_dict = ssig.find_peaks(Phi_hat[z, l, k_range],
                                                prominence=[None, None])
            
            # compensate for k_range starting at k=2
            peaks += 2
            
            # ---------------------------------------------------------------
            # Plot Phi_hat for a given critical freq, time step
            
            # z = 40
            # l = 8
            
            # plt.figure()
            # plt.plot(Phi_hat[z, l, :sb_//2+1], label='Phi hat')
            # plt.plot(peaks, Phi_hat[z, l, peaks], 'r*', label='Peaks')
            # plt.title(f"Critical freq {bark_axis[z]} Bark, time step {time_array[z, l]:.2f} s")
            # plt.legend()
            # ---------------------------------------------------------------
            
            # sort peak indices based on their prominences, from smallest to
            # largest prominence, and pick up to 10 highest
            sort_indices = np.argsort(peaks_dict['prominences'])[-10:]
            
            # get peak indices sorted by increasing prominence
            peaks_sorted = peaks[sort_indices]
            
            # Check if Phi[peak] > 0.05 * max(Phi[all_peaks]) (Eq. 72)
            peak_is_high = (Phi_hat[z, l, peaks_sorted] > 0.05*np.max(Phi_hat[z, l, peaks_sorted]))
            
            # number of peaks in current critical freq, time step
            N_peaks = np.sum(peak_is_high)
            
            # for each tall peak, implement a quadratic fit to estimate modulation
            # rate and amplitude
            f_pi = np.zeros(N_peaks)
            A_pi = np.zeros(N_peaks)
            
            for i, pk in enumerate(peaks_sorted[peak_is_high]):
                
                # solution vector (Eq. 74)
                phi_vec = np.array([Phi_hat[z, l, pk-1],
                                    Phi_hat[z, l, pk],
                                    Phi_hat[z, l, pk+1]])
                
                # modulation index matrix (Eq. 75)
                K = np.array([[(pk-1)**2,   (pk-1), 1],
                              [(pk  )**2,     (pk), 1],
                              [(pk+1)**2,   (pk+1), 1]])
                
                # find vector of unknowns
                C = np.linalg.solve(K, phi_vec)
                
                # first corrected modulation rate
                delta_f_ = fs_/sb_
                f_tilde = -C[1]/(2*C[0]) * delta_f_
                
                # get values of theta, E(theta)
                theta, E = _error_correction()
                
                beta = _beta(theta, f_tilde, delta_f_, E)
                
                theta_min_index = np.argmin(np.abs(beta))
                
                # ------------------------------------------------------------            
                # calculate correction factor following Eq. 78 to 81
                
                # Eq. 80
                theta_min_i = np.argmin(np.abs(beta))
                
                # Eq. 81
                cond1 = (theta[theta_min_i] > 0)
                cond2 = (beta[theta_min_i] * beta[theta_min_i-1] < 0)                
                
                if (cond1 and cond2):
                    tci = theta_min_i       # tci: theta_corr_index
                else:
                    tci = theta_min_i + 1
                
                # Eq. 78 as published - THIS IS WRONG!
                rho_ = (E[tci]
                        - ( (E[tci] - E[tci-1])*
                          beta[tci-1] / (beta[tci] - beta[tci-1])))
                
                # Eq. 78 - corrected
                rho = (E[tci]
                        - ( (E[tci] - E[tci-1])*
                          beta[tci] / (beta[tci] - beta[tci-1])))
                
                
                # # This entire approach is identical to interpolating using
                # # numpy
                # rho_np = np.interp(0, beta, E)
                
                # ------------------------------------------------------------
                # # plot figure comparing the different values for rho
                
                # plt.figure()
                # plt.plot(beta, E, '^:', markersize=10, label='Table 10, Eq. 79')
                # plt.plot(0, rho_np, 'r*', markersize=15, label='np.interp')
                # plt.plot(0, rho_, 'bs', markersize=12, label='Eq. 78 (published)')
                # plt.plot(0, rho, 'mo', markersize=8, label='Eq. 78 (corrected)')
                # plt.grid()
                # plt.xlabel('beta(theta)')
                # plt.ylabel('E(theta)')
                # plt.legend()
                
                # ------------------------------------------------------------
                
                # Corrected modulation rate (Eq. 77)
                f_pi[i] = f_tilde + rho
                
                # Peak amplitudes (Eq. 82)
                pk_range = np.array([pk-1, pk, pk+1])
                A_pi[i] = np.sum(Phi_hat[z, l, pk_range])
                
            
            # ------------------------------------------------------------
            # 7.1.5.2. Weighting of high modulation rates
            
            q1 = 1.2822
            
            # Eq. 87
            if F[z]/1000. < 2**(-3.4253):
                q2 = 0.2471
            else:
                q2 = 0.2471 + 0.0129 * (np.log2(F[z]/1000.) + 3.4253)**2
                
            # 'f_max' is the modulation rate at which the weighting factor G
            # reaches a maximum of 1 (Eq. 86)
            f_max = 72.6937 * (1. - 1.1739*np.exp( -5.4583 * F[z] / 1000. ))
                             
            # Parameters for r_max (Table 11)
            if F[z] < 1000.:
                r1 = 0.3560
                r2 = 0.8049
            else:
                r1 = 0.8024
                r2 = 0.9333
            
            # scaling factor r_max (Eq. 84)
            r_max = 1. / (1. + r1 * np.abs( np.log2(F[z]/1000.) )**r2 )
            
            # weighting factor G (Eq. 85)
            f1 = f_pi/f_max
            f2 = f_max/f_pi
            arg1 = ((f1 - f2) * q1)**2
            
            G = 1. / ((1. + arg1)**q2 )
            
            # Weighted peaks' amplitudes (Eq. 83)
            A_pi_tilde = A_pi * r_max
            A_pi_tilde[ f_pi >= f_max ] *= G[f_pi >= f_max]
            
    
    # 7.1.5.3. Estimation of fundamental modulation rate
    
    # 7.1.5.4. Weighting of low modulation rates
    
    # ************************************************************************
    # Section 7.1.6 of ECMA-418-2, 2nd Ed. (2022)
    
    # Optional entropy weighting based on randomness of the modulation rate
    
    # ************************************************************************
    # Section 7.1.7 of ECMA-418-2, 2nd Ed. (2022)
    
    # Calcuation of time-dependent specific roughness
    
    # ************************************************************************

    
    return bandpass_signals, bark_axis
