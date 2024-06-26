# -*- coding: utf-8 -*-

# Third party imports
import numpy as np

# Local application imports
from mosqito.sound_level_meter import noct_spectrum
from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness
from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
from mosqito.utils import amp2db


def loudness_zwst(signal, fs, field_type="free"):
    """
    Compute the loudness value from a time signal

    This function computes the acoustic loudness according to Zwicker method for
    stationary signals (ISO.532-1:2017).

    Parameters
    ----------
    signal : numpy.array
        Signal time values [Pa], dim (nperseg, nseg).
    fs : float
        Sampling frequency [Hz]
    field_type : str
        Type of soundfield corresponding to spec_third ("free" by
        default or "diffuse").

    Returns
    -------
    N : float or array_like
        Overall loudness array in [sones], size (Ntime,).
    N_specific : array_like
        Specific loudness array [sones/bark], size (Nbark, Ntime).
    bark_axis: array_like
        Bark axis array, size (Nbark,).

    Warning
    -------
    The sampling frequency of the signal must be >= 48 kHz to fulfill requirements.
    If the provided signal doesn't meet the requirements, it will be resampled.

    See Also
    --------
    .loudness_zwst_perseg : Loudness computation by time-segment
    .loudness_zwst_freq : Loudness computation from a sound spectrum
    .loudness_zwtv : Loudness computation for a non-stationary time signal

    Notes
    -----
    The total loudness :math:`N` of the signal is computed as the integral of the specific loudness :math:`N'` measured in sone/bark, over the Bark scale.
    The values of specific loudness are evaluated from third octave band levels as function of critical band rate :math:`z` in Bark.

    .. math::
        N=\\int_{0}^{24Bark}N'(z)\\textup{dz}

    Due to normative continuity, the method is in accordance with ISO 226:1987 equal loudness contours
    instead of ISO 226:2003, as defined in the following standards:
        * ISO 532:1975 (method B)
        * DIN 45631:1991
        * ISO 532-1:2017 (method 1)

    References
    ----------
    :cite:empty:`L_zw-ZF91`
    :cite:empty:`L_zw-ISO.532-1:2017`
    
    .. bibliography::
        :keyprefix: L_zw-

    Examples
    --------
    .. plot::
       :include-source:

       >>> from mosqito.sq_metrics import loudness_zwst
       >>> import matplotlib.pyplot as plt
       >>> import numpy as np
       >>> f=1000
       >>> fs=48000
       >>> d=0.2
       >>> dB=60
       >>> time = np.arange(0, d, 1/fs)
       >>> stimulus = 0.5 * (1 + np.sin(2 * np.pi * f * time))
       >>> rms = np.sqrt(np.mean(np.power(stimulus, 2)))
       >>> ampl = 0.00002 * np.power(10, dB / 20) / rms
       >>> stimulus = stimulus * ampl
       >>> N, N_spec, bark_axis = loudness_zwst(stimulus, fs)
       >>> plt.plot(bark_axis, N_spec)
       >>> plt.xlabel("Frequency band [Bark]")
       >>> plt.ylabel("Specific loudness [Sone/Bark]")
       >>> plt.title("Loudness = " + f"{N:.2f}" + " [Sone]")
    """

    if fs < 48000:
        print(
            "[Warning] Signal resampled to 48 kHz to allow calculation. To fulfill the standard requirements fs should be >=48 kHz."
        )
        from scipy.signal import resample

        signal = resample(signal, int(48000 * len(signal) / fs))
        fs = 48000
    # Compute third octave band spectrum
    spec_third, _ = noct_spectrum(signal, fs, fmin=24, fmax=12600)

    # Compute dB values
    spec_third = amp2db(spec_third, ref=2e-5)

    # Compute main loudness
    Nm = _main_loudness(spec_third, field_type)

    # Computation of specific loudness pattern and integration of overall
    # loudness by attaching slopes towards higher frequencies
    N, N_specific = _calc_slopes(Nm)

    # Define Bark axis
    bark_axis = np.linspace(0.1, 24, int(24 / 0.1))

    return N, N_specific, bark_axis
