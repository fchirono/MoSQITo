import numpy as np

def fm_sine_generator(xmod, fs, fc, k, spl_level, print_info=False):
    """
    Creates a frequency-modulated (FM) signal of level 'spl_level' (in dB SPL)
    with sinusoidal carrier of frequency 'fc', arbitrary modulating signal
    'xm', frequency sensitivity 'k', and sampling frequency 'fs'. The FM signal
    length is the same as the length of 'xm'. 
    
    Parameters
    ----------
    xmod: array
        Modulating signal, dim(N)
    fs: float
        Sampling frequency, in [Hz].
    fc: float
        Carrier frequency, in [Hz]. Must be less than 'fs/2'.
    k: float
        Frequency sensitivity of the modulator. 
    spl_level: float
        Sound Pressure Level [dB ref 20 uPa RMS] of the modulated signal.
    print_info: bool, optional
        If True, the maximum frequency deviation and modulation index are printed. 
        Default is False
    
    Returns
    -------
    y_fm: numpy.array
        Frequency-modulated signal with sine carrier, dim(N) in [Pa].
    inst_freq: numpy.array
        Instantaneaous frequency, dim(N)
    max_freq_deviation: float
        Maximum frequency deviation [Hz]   
    FM_modulation_index: float
        Modulation index 
        
    Warning
    -------
    spl_level must be provided in dB, ref=2e-5 Pa.
    
    Notes
    -----
    The frequency sensitivity 'k' is equal to the frequency deviation in Hz 
    away from 'fc' per unit amplitude of the modulating signal 'xmod'.
        
    Examples
    --------
    .. plot::
       :include-source:
       
        >>> from mosqito.utils import fm_sine_generator
        >>> import matplotlib.pyplot as plt
        >>> fs = 48000     # [Hz]
        >>> duration = 1
        >>> t = np.linspace(0, duration, int(fs*duration))
        >>> dB = 60        # [dB SPL]
        >>> fc = 50        # [Hz]
        >>> k = fc//2      # [Hz per unit amplitude of 'xm']
        >>> fm = 10        # [Hz]
        >>> xmod = np.sin(2*np.pi*t*fm)
        >>> y_fm, inst_freq, f_delta, m  = fm_sine_generator(xmod, fs, fc, k, dB, True)
        >>> plt.figure()
        >>> plt.plot(t, y_fm)
        >>> plt.xlabel("Time axis [s]")
        >>> plt.ylabel("Amplitude signal [Pa]")    
        >>> plt.title(f'Modulation index = {m:.1f}')
        >>> plt.figure()
        >>> plt.xlabel("Time axis [s]")
        >>> plt.ylabel("Instantaneous frequency [Hz]")    
        >>> plt.title(f'Max frequency deviation = {f_delta:.1f}')
        >>> plt.plot(t, inst_freq)
        
    """
    
    assert fc < fs/2, "Carrier frequency 'fc' must be less than 'fs/2'!"
    
     # sampling interval in seconds
    dt = 1/fs

    # instantaneous frequency of FM signal
    inst_freq = fc + k*xmod
    
    # unit-amplitude FM signal
    y_fm = np.sin(2*np.pi * np.cumsum(inst_freq)*dt)
    
    # max frequency deviation
    f_delta = k * np.max(np.abs(xmod))
    
    # FM modulation index
    m = np.max(np.abs(2*np.pi * k * np.cumsum(xmod)*dt))
    
    # Apply amplitude factor to obtain 'spl_level'
    p_ref = 20e-6
    A_rms = p_ref * 10**(spl_level/20)
    y_fm *= A_rms/np.std(y_fm)

    if print_info:
        print(f'\tMax freq deviation: {f_delta} Hz')
        print(f'\tFM modulation index: {m:.2f}')

    return y_fm, inst_freq, f_delta, m
