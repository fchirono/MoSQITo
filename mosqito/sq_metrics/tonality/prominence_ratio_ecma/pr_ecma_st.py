# -*- coding: utf-8 -*-

# Local imports
from mosqito.sound_level_meter.comp_spectrum import comp_spectrum
from mosqito.sq_metrics.tonality.prominence_ratio_ecma._pr_main_calc import _pr_main_calc


def pr_ecma_st(signal, fs, prominence=True):
    """
    Compute the tone-to-noise ratio value from a time signal

    This function computes the prominence ratio according to ECMA-418-1
    for a stationary signal.

    Parameters
    ----------
    signal :numpy.array
        Signal time values in [Pa].
    fs : integer
        Sampling frequency.
    prominence : Bool
        If True, the algorithm only returns the prominent tones, if False it returns all tones detected.
        Default to True

    Returns
    -------
    t_pr : float
        Global PR value.
    pr : array of float
        PR values for each detected tone.
    promi : array of bool
        Prominence criterion for each detected tone.
    tones_freqs : array of float
        Frequency of the detected tones.

    See Also
    --------
    .pr_ecma_freq : PR computation for a sound spectrum
    .pr_ecma_perseg : PR computation for a non-stationary signal
    .tnr_ecma_st : Tone-to-noise ratio for a stationary signal

    Notes
    -----
    The computation is based on a spectrum analysis detecting peaks to be compared with the overall smoothed spectrum.
    The algorithm automatically detects the frequency of the tonal components according to Sottek's method.

    .. math::
        \\Delta L_{TNR} = L_{peak} - 10\\log_{10}\\left (10^{0.1L_{peakband}} -10^{0.1L_{peak}}\\right )

    .. math::
        \\Delta L_{PR} = 10\\log_{10}\\left ( 10^{0.1L_{peakband}} \\right ) - 10\\log_{10}\\left [0.5\\left (10^{0.1L_{lowerband}} -10^{0.1L_{upperband}}\\right )\\right]

    The difference between PR and TNR lies in the comparison process between the peak level and the background noise amplitude.
    TNR compares the peak level to the level of its critical band, while PR compares the level of the peak's critical band to its two neighbor bands.
    According to ECMA 418-1 standard, TNR can then prove to be more accurate for multiple tones in adjacent critical bands, for example when strong harmonics exist.
    PR can be more effective for multiple tones within the same critical band and is more readily automated to handle such cases.

    Along with the TNR/PR value comes a prominence indicator, a tone being considered as prominent if its dB level is sufficiently higher than the smoothed spectrum, depending on its frequency.

    References
    ----------
    :cite:empty:`PR-ECMA-418-2`

    .. bibliography::
        :keyprefix: PR-

    Examples
    --------
    The example stimulus is made of white noise + 2 sine waves at 1kHz and 3kHz.

    .. plot::
       :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mosqito.sound_level_meter import comp_spectrum
        >>> fs = 48000
        >>> d = 2
        >>> f = 1000
        >>> dB = 60
        >>> time = np.arange(0, d, 1/fs)
        >>> stimulus = np.sin(2 * np.pi * f * time) + 0.5 * np.sin(2 * np.pi * 3 * f * time)+ np.random.normal(0,0.5, len(time))
        >>> rms = np.sqrt(np.mean(np.power(stimulus, 2)))
        >>> ampl = 0.00002 * np.power(10, dB / 20) / rms
        >>> stimulus = stimulus * ampl
        >>> spectrum_db, freq_axis = comp_spectrum(stimulus, fs, db=True)
        >>> plt.plot(freq_axis, spectrum_db)
        >>> plt.ylim(0,60)
        >>> plt.xlabel("Frequency [Hz]")
        >>> plt.ylabel("Acoustic pressure [dB]")

    .. plot::
       :include-source:

        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from mosqito.sq_metrics import pr_ecma_st
        >>> fs = 48000
        >>> d = 2
        >>> f = 1000
        >>> dB = 60
        >>> time = np.arange(0, d, 1/fs)
        >>> stimulus = np.sin(2 * np.pi * f * time) + 0.5 * np.sin(2 * np.pi * 3 * f * time)+ np.random.normal(0,0.5, len(time))
        >>> rms = np.sqrt(np.mean(np.power(stimulus, 2)))
        >>> ampl = 0.00002 * np.power(10, dB / 20) / rms
        >>> stimulus = stimulus * ampl
        >>> t_pr, pr, prom, tones_freqs = pr_ecma_st(stimulus, fs)
        >>> plt.bar(tones_freqs, pr, width=50)
        >>> plt.grid(axis='y')
        >>> plt.ylabel("PR [dB]")
        >>> plt.title("Total PR = "+ f"{t_pr[0]:.2f}" + " dB")
        >>> plt.xscale('log')
        >>> xticks_pos = list(tones_freqs) + [100,1000,10000]
        >>> xticks_pos = np.sort(xticks_pos)
        >>> xticks_label = [str(elem) for elem in xticks_pos]
        >>> plt.xticks(xticks_pos, labels=xticks_label, rotation = 30)
        >>> plt.xlabel("Frequency [Hz]")
    """

    # Compute db spectrum
    spectrum_db, freq_axis = comp_spectrum(signal, fs, db=True)

    # Compute PR values
    tones_freqs, pr, prom, t_pr = _pr_main_calc(spectrum_db, freq_axis)
    prom = prom.astype(bool)

    if prominence == False:
        return t_pr, pr, prom, tones_freqs
    else:
        return t_pr, pr[prom], prom[prom], tones_freqs[prom]
