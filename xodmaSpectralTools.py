# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodmaSpectralTools.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
# XODMK Audio Tools - Spectral processing functions
#
# STFT, Phase Vocoder
#
# requires: odmkSpectralUtil.py, 
#       
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************


import sys
import warnings
import numpy as np
import scipy.fftpack as fft
import scipy
import scipy.signal
import scipy.interpolate
import six


rootDir = '../../'
sys.path.insert(0, rootDir+'audio/xodma')

# import cache
from cache import cache
from xodmaSpectralUtil import MAX_MEM_BLOCK, note_to_hz, hz_to_midi, hz_to_octs
from xodmaSpectralUtil import fft_frequencies, mel_frequencies, A_weighting
from xodmaSpectralUtil import frame, get_window, window_bandwidth
from xodmaMiscUtil import tiny, fix_length, valid_int, normalize, pad_center
from xodmaAudioTools import valid_audio, resample
from xodmaParameterError import ParameterError


# temp python debugger - use >>>pdb.set_trace() to set break
#import pdb



__all__ = ['mag_spectrogram',
           'stft', 'istft', 'magphase', 'iirt',
           'ifgram', 'phase_vocoder',
           'perceptual_weighting',
           'power_to_db', 'db_to_power',
           'amplitude_to_db', 'db_to_amplitude',
           'fmt',
           'dct',
           'mel',
           'chroma',
           'constant_q',
           'constant_q_lengths',
           'cq_to_chroma',
           'peak_pick']
           
           
# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# helper function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def mag_spectrogram(y=None, S=None, n_fft=2048, hop_length=512, power=1):
    '''Helper function to retrieve a magnitude spectrogram.

    This is primarily used in feature extraction functions that can operate on
    either audio time-series or spectrogram input.


    Parameters
    ----------
    y : None or np.ndarray [ndim=1]
        If provided, an audio time series

    S : None or np.ndarray
        Spectrogram input, optional

    n_fft : int > 0
        STFT window size

    hop_length : int > 0
        STFT hop length

    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.

    Returns
    -------
    S_out : np.ndarray [dtype=np.float32]
        - If `S` is provided as input, then `S_out == S`
        - Else, `S_out = |stft(y, n_fft=n_fft, hop_length=hop_length)|**power`

    n_fft : int > 0
        - If `S` is provided, then `n_fft` is inferred from `S`
        - Else, copied from input
    '''

    if S is not None:
        # Infer n_fft from spectrogram shape
        n_fft = 2 * (S.shape[0] - 1)
    else:
        # Otherwise, compute a magnitude spectrogram from input
        S = np.abs(stft(y, n_fft=n_fft, hop_length=hop_length))**power

    return S, n_fft



# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# Main function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


@cache(level=20)
def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
    """Short-time Fourier transform (STFT)

    Returns a complex-valued matrix D such that
        `np.abs(D[f, t])` is the magnitude of frequency bin `f`
        at frame `t`

        `np.angle(D[f, t])` is the phase of frequency bin `f`
        at frame `t`

    Parameters
    ----------
    y : np.ndarray [shape=(n,)], real-valued
        the input signal (audio time series)

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        number of audio samples between STFT columns.
        If unspecified, defaults `win_length / 4`.

    win_length  : int <= n_fft [scalar]
        Each frame of audio is windowed by `window()`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`.

        If unspecified, defaults to ``win_length = n_fft``.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a vector or array of length `n_fft`

        .. see also:: `filters.get_window`

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    dtype       : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.

    mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.


    Returns
    -------
    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=dtype]
        STFT matrix


    See Also
    --------
    istft : Inverse STFT

    ifgram : Instantaneous frequency spectrogram

    np.pad : array padding

    Notes
    -----
    This function caches at level 20.


    Examples
    --------

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> D
    array([[  2.576e-03 -0.000e+00j,   4.327e-02 -0.000e+00j, ...,
              3.189e-04 -0.000e+00j,  -5.961e-06 -0.000e+00j],
           [  2.441e-03 +2.884e-19j,   5.145e-02 -5.076e-03j, ...,
             -3.885e-04 -7.253e-05j,   7.334e-05 +3.868e-04j],
          ...,
           [ -7.120e-06 -1.029e-19j,  -1.951e-09 -3.568e-06j, ...,
             -4.912e-07 -1.487e-07j,   4.438e-06 -1.448e-05j],
           [  7.136e-06 -0.000e+00j,   3.561e-06 -0.000e+00j, ...,
             -5.144e-07 -0.000e+00j,  -1.514e-05 -0.000e+00j]], dtype=complex64)


    Use left-aligned frames, instead of centered frames
    >>> D_left = librosa.stft(y, center=False)

    Use a shorter hop length
    >>> D_short = librosa.stft(y, hop_length=64)

    Display a spectrogram

    >>> import matplotlib.pyplot as plt
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.title('Power spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()

    """

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        valid_audio(y)
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # *** uses numpy 'as_stride' => creates y_frames as a new memory 'view' 
    #     interprets 1D memory structure y as 'virtual overlapped memory structure'
    #     doesn't require any additional memory allocation ***
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)
    
    # inspect AmenB @ 1024 - y_frames.shape => (1024, 652)
    
    # inspect y frame[0]: max(y_frames[:,0] - y[0:1024]) = 0
    # inspect y frame[1]: max(y_frames[:,1] - y[256:1024+256]) = 0
    # inspect y frame[2]: max(y_frames[:,2] - y[512:1024+512]) = 0
    
    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                            dtype=dtype, order='F')
    
    
    # inspect AmenB @ 1024 - stft_matrix.shape => (513, 652)

    # how many columns can we fit within MAX_MEM_BLOCK?
    # ** note: pymalloc allocates memory in 256 kB chunks, called arenas
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] * stft_matrix.itemsize))
    
    # inspect AmenB @ 1024 - n_columns = 63

    for memBlk_s in range(0, stft_matrix.shape[1], n_columns):
    #print(list(range(0,652,63))) [0, 63, 126, 189, 252, 315, 378, 441, 504, 567, 630]
    
        # limits final 256 KB 'Arena' mem block to final input sample
        memLast_s = min(memBlk_s + n_columns, stft_matrix.shape[1])


        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, memBlk_s:memLast_s] = fft.fft(fft_window *
                                             y_frames[:, memBlk_s:memLast_s],
                                             axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix



@cache(level=30)
def istft(stft_matrix, hop_length=None, win_length=None, window='hann',
          center=True, dtype=np.float32, length=None):
    """
    Inverse short-time Fourier transform (ISTFT).

    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_.

    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.

    .. [1] D. W. Griffin and J. S. Lim,
        "Signal estimation from modified short-time Fourier transform,"
        IEEE Trans. ASSP, vol.32, no.2, pp.236–243, Apr. 1984.

    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`

    hop_length  : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.

    win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).

        If unspecified, defaults to `n_fft`.

    window      : string, tuple, number, function, np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, or number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

        .. see also:: `filters.get_window`

    center      : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.

    dtype       : numeric type
        Real numeric type for `y`.  Default is 32-bit float.

    length : int > 0, optional
        If provided, the output `y` is zero-padded or clipped to exactly
        `length` samples.

    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`

    See Also
    --------
    stft : Short-time Fourier Transform

    Notes
    -----
    This function caches at level 30.

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)

    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of `y` will not be represented.

    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.istft(D, length=n)
    >>> np.max(np.abs(y - y_out))
    1.4901161e-07
    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    ifft_window = get_window(window, win_length, fftbins=True)

    # Pad out to match n_fft
    ifft_window = pad_center(ifft_window, n_fft)

    n_frames = stft_matrix.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_sum = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_square = ifft_window * ifft_window

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
        ifft_window_sum[sample:(sample + n_fft)] += ifft_window_square

    # Normalize by sum of squared window
    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)
    y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if length is None:
        # If we don't need to control length, just do the usual center trimming
        # to eliminate padded data
        if center:
            y = y[int(n_fft // 2):-int(n_fft // 2)]
    else:
        if center:
            # If we're centering, crop off the first n_fft//2 samples
            # and then trim/pad to the target length.
            # We don't trim the end here, so that if the signal is zero-padded
            # to a longer duration, the decay is smooth by windowing
            start = int(n_fft // 2)
        else:
            # If we're not centering, start at 0 and trim/pad as necessary
            start = 0

        y = fix_length(y[start:], length)

    return y


def ifgram(y, sr=22050, n_fft=2048, hop_length=None, win_length=None,
           window='hann', norm=False, center=True, ref_power=1e-6,
           clip=True, dtype=np.complex64, pad_mode='reflect'):
    '''Compute the instantaneous frequency (as a proportion of the sampling rate)
    obtained as the time-derivative of the phase of the complex spectrum as
    described by [1]_.

    Calculates regular STFT as a side effect.

    .. [1] Abe, Toshihiko, Takao Kobayashi, and Satoshi Imai.
        "Harmonics tracking and pitch extraction based on instantaneous
        frequency."
        International Conference on Acoustics, Speech, and Signal Processing,
        ICASSP-95., Vol. 1. IEEE, 1995.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    n_fft : int > 0 [scalar]
        FFT window size

    hop_length : int > 0 [scalar]
        hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.

    win_length : int > 0, <= n_fft
        Window length. Defaults to `n_fft`.
        See `stft` for details.

    window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
        - a window specification (string, tuple, number);
          see `scipy.signal.get_window`
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`

        See `stft` for details.

        .. see also:: `filters.get_window`

    norm : bool
        Normalize the STFT.

    center      : boolean
        - If `True`, the signal `y` is padded so that frame
            `D[:, t]` (and `if_gram`) is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` at `y[t * hop_length]`

    ref_power : float >= 0 or callable
        Minimum power threshold for estimating instantaneous frequency.
        Any bin with `np.abs(D[f, t])**2 < ref_power` will receive the
        default frequency estimate.

        If callable, the threshold is set to `ref_power(np.abs(D)**2)`.

    clip : boolean
        - If `True`, clip estimated frequencies to the range `[0, 0.5 * sr]`.
        - If `False`, estimated frequencies can be negative or exceed
          `0.5 * sr`.

    dtype : numeric type
        Complex numeric type for `D`.  Default is 64-bit complex.

    mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, STFT uses reflection padding.


    Returns
    -------
    if_gram : np.ndarray [shape=(1 + n_fft/2, t), dtype=real]
        Instantaneous frequency spectrogram:
        `if_gram[f, t]` is the frequency at bin `f`, time `t`

    D : np.ndarray [shape=(1 + n_fft/2, t), dtype=complex]
        Short-time Fourier transform

    See Also
    --------
    stft : Short-time Fourier Transform

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> frequencies, D = librosa.ifgram(y, sr=sr)
    >>> frequencies
    array([[  0.000e+00,   0.000e+00, ...,   0.000e+00,   0.000e+00],
           [  3.150e+01,   3.070e+01, ...,   1.077e+01,   1.077e+01],
           ...,
           [  1.101e+04,   1.101e+04, ...,   1.101e+04,   1.101e+04],
           [  1.102e+04,   1.102e+04, ...,   1.102e+04,   1.102e+04]])

    '''

    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # Construct a padded hann window
    fft_window = pad_center(get_window(window, win_length, fftbins=True),
                                 n_fft)

    # Window for discrete differentiation
    freq_angular = np.linspace(0, 2 * np.pi, n_fft, endpoint=False)

    d_window = np.sin(-freq_angular) * np.pi / n_fft

    stft_matrix = stft(y, n_fft=n_fft, hop_length=hop_length,
                       win_length=win_length,
                       window=window, center=center,
                       dtype=dtype, pad_mode=pad_mode)

    diff_stft = stft(y, n_fft=n_fft, hop_length=hop_length,
                     window=d_window, center=center,
                     dtype=dtype, pad_mode=pad_mode).conj()

    # Compute power normalization. Suppress zeros.
    mag, phase = magphase(stft_matrix)

    if six.callable(ref_power):
        ref_power = ref_power(mag**2)
    elif ref_power < 0:
        raise ParameterError('ref_power must be non-negative or callable.')

    # Pylint does not correctly infer the type here, but it's correct.
    # pylint: disable=maybe-no-member
    freq_angular = freq_angular.reshape((-1, 1))
    bin_offset = (-phase * diff_stft).imag / mag

    bin_offset[mag < ref_power**0.5] = 0

    if_gram = freq_angular[:n_fft//2 + 1] + bin_offset

    if norm:
        stft_matrix = stft_matrix * 2.0 / fft_window.sum()

    if clip:
        np.clip(if_gram, 0, np.pi, out=if_gram)

    if_gram *= float(sr) * 0.5 / np.pi

    return if_gram, stft_matrix


def magphase(D, power=1):
    """Separate a complex-valued spectrogram D into its magnitude (S)
    and phase (P) components, so that `D = S * P`.


    Parameters
    ----------
    D       : np.ndarray [shape=(d, t), dtype=complex]
        complex-valued spectrogram
    power : float > 0
        Exponent for the magnitude spectrogram,
        e.g., 1 for energy, 2 for power, etc.


    Returns
    -------
    D_mag   : np.ndarray [shape=(d, t), dtype=real]
        magnitude of `D`, raised to `power`
    D_phase : np.ndarray [shape=(d, t), dtype=complex]
        `exp(1.j * phi)` where `phi` is the phase of `D`


    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> magnitude, phase = librosa.magphase(D)
    >>> magnitude
    array([[  2.524e-03,   4.329e-02, ...,   3.217e-04,   3.520e-05],
           [  2.645e-03,   5.152e-02, ...,   3.283e-04,   3.432e-04],
           ...,
           [  1.966e-05,   9.828e-06, ...,   3.164e-07,   9.370e-06],
           [  1.966e-05,   9.830e-06, ...,   3.161e-07,   9.366e-06]], dtype=float32)
    >>> phase
    array([[  1.000e+00 +0.000e+00j,   1.000e+00 +0.000e+00j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j],
           [  1.000e+00 +1.615e-16j,   9.950e-01 -1.001e-01j, ...,
              9.794e-01 +2.017e-01j,   1.492e-02 -9.999e-01j],
           ...,
           [  1.000e+00 -5.609e-15j,  -5.081e-04 +1.000e+00j, ...,
             -9.549e-01 -2.970e-01j,   2.938e-01 -9.559e-01j],
           [ -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j, ...,
             -1.000e+00 +8.742e-08j,  -1.000e+00 +8.742e-08j]], dtype=complex64)


    Or get the phase angle (in radians)

    >>> np.angle(phase)
    array([[  0.000e+00,   0.000e+00, ...,   3.142e+00,   3.142e+00],
           [  1.615e-16,  -1.003e-01, ...,   2.031e-01,  -1.556e+00],
           ...,
           [ -5.609e-15,   1.571e+00, ...,  -2.840e+00,  -1.273e+00],
           [  3.142e+00,   3.142e+00, ...,   3.142e+00,   3.142e+00]], dtype=float32)

    """

    mag = np.abs(D)
    mag **= power
    phase = np.exp(1.j * np.angle(D))

    return mag, phase


def phase_vocoder(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`

    Based on the implementation provided by [1]_.

    .. [1] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    Examples
    --------
    >>> # Play at double speed
    >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
    >>> y_fast  = librosa.istft(D_fast, hop_length=512)

    >>> # Or play at 1/3 speed
    >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    >>> D_slow  = librosa.phase_vocoder(D, 1./3, hop_length=512)
    >>> y_slow  = librosa.istft(D_slow, hop_length=512)

    Parameters
    ----------
    D : np.ndarray [shape=(d, t), dtype=complex]
        STFT matrix

    rate :  float > 0 [scalar]
        Speed-up factor: `rate > 1` is faster, `rate < 1` is slower.

    hop_length : int > 0 [scalar] or None
        The number of samples between successive columns of `D`.

        If None, defaults to `n_fft/4 = (D.shape[0]-1)/2`

    Returns
    -------
    D_stretched  : np.ndarray [shape=(d, t / rate), dtype=complex]
        time-stretched STFT
    """

    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, D.shape[1], rate, dtype=np.float)

    # Create an empty output array
    d_stretch = np.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = np.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):

        columns = D[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = ((1.0 - alpha) * np.abs(columns[:, 0])
               + alpha * np.abs(columns[:, 1]))

        # Store to output array
        d_stretch[:, t] = mag * np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (np.angle(columns[:, 1])
                  - np.angle(columns[:, 0])
                  - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch


@cache(level=20)
def iirt(y, sr=22050, win_length=2048, hop_length=None, center=True,
         tuning=0.0, pad_mode='reflect', **kwargs):
    r'''Time-frequency representation using IIR filters [1]_.

    This function will return a time-frequency representation
    using a multirate filter bank consisting of IIR filters.
    First, `y` is resampled as needed according to the provided `sample_rates`.
    Then, a filterbank with with `n` band-pass filters is designed.
    The resampled input signals are processed by the filterbank as a whole.
    (`scipy.signal.filtfilt` is used to make the phase linear.)
    The output of the filterbank is cut into frames.
    For each band, the short-time mean-square power (STMSP) is calculated by
    summing `win_length` subsequent filtered time samples.

    When called with the default set of parameters, it will generate the TF-representation
    as described in [1]_ (pitch filterbank):

        * 85 filters with MIDI pitches [24, 108] as `center_freqs`.
        * each filter having a bandwith of one semitone.

    .. [1] Müller, Meinard.
           "Information Retrieval for Music and Motion."
           Springer Verlag. 2007.


    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time series

    sr : number > 0 [scalar]
        sampling rate of `y`

    win_length : int > 0, <= n_fft
        Window length.

    hop_length : int > 0 [scalar]
        Hop length, number samples between subsequent frames.
        If not supplied, defaults to `win_length / 4`.

    center : boolean
        - If `True`, the signal `y` is padded so that frame
          `D[:, t]` is centered at `y[t * hop_length]`.
        - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin.

    pad_mode : string
        If `center=True`, the padding mode to use at the edges of the signal.
        By default, this function uses reflection padding.

    kwargs : additional keyword arguments
        Additional arguments for `librosa.filters.semitone_filterbank()`
        (e.g., could be used to provide another set of `center_freqs` and `sample_rates`).

    Returns
    -------
    bands_power : np.ndarray [shape=(n, t), dtype=dtype]
        Short-time mean-square power for the input signal.

    See Also
    --------
    librosa.filters.semitone_filterbank
    librosa.filters._multirate_fb
    librosa.filters.mr_frequencies
    librosa.core.cqt
    scipy.signal.filtfilt

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.iirt(y)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='cqt_hz', x_axis='time')
    >>> plt.title('Semitone spectrogram')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    '''

    # check audio input
    valid_audio(y)

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(hop_length), mode=pad_mode)

    # get the semitone filterbank
    filterbank_ct, sample_rates = semitone_filterbank(tuning=tuning, **kwargs)

    # create three downsampled versions of the audio signal
    y_resampled = []

    y_srs = np.unique(sample_rates)

    for cur_sr in y_srs:
        y_resampled.append(resample(y, sr, cur_sr))

    # Compute the number of frames that will fit. The end may get truncated.
    n_frames = 1 + int((len(y) - win_length) / float(hop_length))

    bands_power = []

    for cur_sr, cur_filter in zip(sample_rates, filterbank_ct):
        factor = float(sr) / float(cur_sr)
        win_length_STMSP = int(np.round(win_length / factor))
        hop_length_STMSP = int(np.round(hop_length / factor))

        # filter the signal
        cur_sr_idx = np.flatnonzero(y_srs == cur_sr)[0]

        cur_filter_output = scipy.signal.filtfilt(cur_filter[0], cur_filter[1],
                                                  y_resampled[cur_sr_idx])

        # frame the current filter output
        cur_frames = frame(np.ascontiguousarray(cur_filter_output),
                                frame_length=win_length_STMSP,
                                hop_length=hop_length_STMSP)

        bands_power.append(factor * np.sum(cur_frames**2, axis=0)[:n_frames])

    return np.asarray(bands_power)


@cache(level=30)
def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    """Convert a power spectrogram (amplitude squared) to decibel (dB) units

    This computes the scaling ``10 * log10(S / ref)`` in a numerically
    stable way.

    Parameters
    ----------
    S : np.ndarray
        input power

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `10 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `abs(S)` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(10 * log10(S)) - top_db``

    Returns
    -------
    S_db   : np.ndarray
        ``S_db ~= 10 * log10(S) - 10 * log10(ref)``

    See Also
    --------
    perceptual_weighting
    db_to_power
    amplitude_to_db
    db_to_amplitude

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    Get a power spectrogram from a waveform ``y``

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> librosa.power_to_db(S**2)
    array([[-33.293, -27.32 , ..., -33.293, -33.293],
           [-33.293, -25.723, ..., -33.293, -33.293],
           ...,
           [-33.293, -33.293, ..., -33.293, -33.293],
           [-33.293, -33.293, ..., -33.293, -33.293]], dtype=float32)

    Compute dB relative to peak power

    >>> librosa.power_to_db(S**2, ref=np.max)
    array([[-80.   , -74.027, ..., -80.   , -80.   ],
           [-80.   , -72.431, ..., -80.   , -80.   ],
           ...,
           [-80.   , -80.   , ..., -80.   , -80.   ],
           [-80.   , -80.   , ..., -80.   , -80.   ]], dtype=float32)


    Or compare to median power

    >>> librosa.power_to_db(S**2, ref=np.median)
    array([[-0.189,  5.784, ..., -0.189, -0.189],
           [-0.189,  7.381, ..., -0.189, -0.189],
           ...,
           [-0.189, -0.189, ..., -0.189, -0.189],
           [-0.189, -0.189, ..., -0.189, -0.189]], dtype=float32)


    And plot the results

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(S**2, sr=sr, y_axis='log')
    >>> plt.colorbar()
    >>> plt.title('Power spectrogram')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(librosa.power_to_db(S**2, ref=np.max),
    ...                          sr=sr, y_axis='log', x_axis='time')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.title('Log-Power spectrogram')
    >>> plt.tight_layout()

    """

    S = np.asarray(S)

    if amin <= 0:
        raise ParameterError('amin must be strictly positive')

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('power_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call power_to_db(magphase(D, power=2)[0]) instead.')
        magnitude = np.abs(S)
    else:
        magnitude = S

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ParameterError('top_db must be non-negative')
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


@cache(level=30)
def db_to_power(S_db, ref=1.0):
    '''Convert a dB-scale spectrogram to a power spectrogram.

    This effectively inverts `power_to_db`:

        `db_to_power(S_db) ~= ref * 10.0**(S_db / 10)`

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram

    ref : number > 0
        Reference power: output will be scaled by this value

    Returns
    -------
    S : np.ndarray
        Power spectrogram

    Notes
    -----
    This function caches at level 30.
    '''
    return ref * np.power(10.0, 0.1 * S_db)


@cache(level=30)
def amplitude_to_db(S, ref=1.0, amin=1e-5, top_db=80.0):
    '''Convert an amplitude spectrogram to dB-scaled spectrogram.

    This is equivalent to ``power_to_db(S**2)``, but is provided for convenience.

    Parameters
    ----------
    S : np.ndarray
        input amplitude

    ref : scalar or callable
        If scalar, the amplitude `abs(S)` is scaled relative to `ref`:
        `20 * log10(S / ref)`.
        Zeros in the output correspond to positions where `S == ref`.

        If callable, the reference value is computed as `ref(S)`.

    amin : float > 0 [scalar]
        minimum threshold for `S` and `ref`

    top_db : float >= 0 [scalar]
        threshold the output at `top_db` below the peak:
        ``max(20 * log10(S)) - top_db``


    Returns
    -------
    S_db : np.ndarray
        ``S`` measured in dB

    See Also
    --------
    power_to_db, db_to_amplitude

    Notes
    -----
    This function caches at level 30.
    '''

    S = np.asarray(S)

    if np.issubdtype(S.dtype, np.complexfloating):
        warnings.warn('amplitude_to_db was called on complex input so phase '
                      'information will be discarded. To suppress this warning, '
                      'call amplitude_to_db(magphase(D)[0]) instead.')

    magnitude = np.abs(S)

    if six.callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    power = np.square(magnitude, out=magnitude)

    return power_to_db(power, ref=ref_value**2, amin=amin**2,
                       top_db=top_db)


@cache(level=30)
def db_to_amplitude(S_db, ref=1.0):
    '''Convert a dB-scaled spectrogram to an amplitude spectrogram.

    This effectively inverts `amplitude_to_db`:

        `db_to_amplitude(S_db) ~= 10.0**(0.5 * (S_db + log10(ref)/10))`

    Parameters
    ----------
    S_db : np.ndarray
        dB-scaled spectrogram

    ref: number > 0
        Optional reference power.

    Returns
    -------
    S : np.ndarray
        Linear magnitude spectrogram

    Notes
    -----
    This function caches at level 30.
    '''
    return db_to_power(S_db, ref=ref**2)**0.5


@cache(level=30)
def perceptual_weighting(S, frequencies, **kwargs):
    '''Perceptual weighting of a power spectrogram:

    `S_p[f] = A_weighting(f) + 10*log(S[f] / ref)`

    Parameters
    ----------
    S : np.ndarray [shape=(d, t)]
        Power spectrogram

    frequencies : np.ndarray [shape=(d,)]
        Center frequency for each row of `S`

    kwargs : additional keyword arguments
        Additional keyword arguments to `power_to_db`.

    Returns
    -------
    S_p : np.ndarray [shape=(d, t)]
        perceptually weighted version of `S`

    See Also
    --------
    power_to_db

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    Re-weight a CQT power spectrum, using peak power as reference

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> CQT = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A1'))
    >>> freqs = librosa.cqt_frequencies(CQT.shape[0],
    ...                                 fmin=librosa.note_to_hz('A1'))
    >>> perceptual_CQT = librosa.perceptual_weighting(CQT**2,
    ...                                               freqs,
    ...                                               ref=np.max)
    >>> perceptual_CQT
    array([[ -80.076,  -80.049, ..., -104.735, -104.735],
           [ -78.344,  -78.555, ..., -103.725, -103.725],
           ...,
           [ -76.272,  -76.272, ...,  -76.272,  -76.272],
           [ -76.485,  -76.485, ...,  -76.485,  -76.485]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(CQT,
    ...                                                  ref=np.max),
    ...                          fmin=librosa.note_to_hz('A1'),
    ...                          y_axis='cqt_hz')
    >>> plt.title('Log CQT power')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.subplot(2, 1, 2)
    >>> librosa.display.specshow(perceptual_CQT, y_axis='cqt_hz',
    ...                          fmin=librosa.note_to_hz('A1'),
    ...                          x_axis='time')
    >>> plt.title('Perceptually weighted log CQT')
    >>> plt.colorbar(format='%+2.0f dB')
    >>> plt.tight_layout()
    '''

    offset = A_weighting(frequencies).reshape((-1, 1))

    return offset + power_to_db(S, **kwargs)


@cache(level=30)
def fmt(y, t_min=0.5, n_fmt=None, kind='cubic', beta=0.5, over_sample=1, axis=-1):
    """The fast Mellin transform (FMT) [1]_ of a uniformly sampled signal y.

    When the Mellin parameter (beta) is 1/2, it is also known as the scale transform [2]_.
    The scale transform can be useful for audio analysis because its magnitude is invariant
    to scaling of the domain (e.g., time stretching or compression).  This is analogous
    to the magnitude of the Fourier transform being invariant to shifts in the input domain.


    .. [1] De Sena, Antonio, and Davide Rocchesso.
        "A fast Mellin and scale transform."
        EURASIP Journal on Applied Signal Processing 2007.1 (2007): 75-75.

    .. [2] Cohen, L.
        "The scale representation."
        IEEE Transactions on Signal Processing 41, no. 12 (1993): 3275-3292.

    Parameters
    ----------
    y : np.ndarray, real-valued
        The input signal(s).  Can be multidimensional.
        The target axis must contain at least 3 samples.

    t_min : float > 0
        The minimum time spacing (in samples).
        This value should generally be less than 1 to preserve as much information as
        possible.

    n_fmt : int > 2 or None
        The number of scale transform bins to use.
        If None, then `n_bins = over_sample * ceil(n * log((n-1)/t_min))` is taken,
        where `n = y.shape[axis]`

    kind : str
        The type of interpolation to use when re-sampling the input.
        See `scipy.interpolate.interp1d` for possible values.

        Note that the default is to use high-precision (cubic) interpolation.
        This can be slow in practice; if speed is preferred over accuracy,
        then consider using `kind='linear'`.

    beta : float
        The Mellin parameter.  `beta=0.5` provides the scale transform.

    over_sample : float >= 1
        Over-sampling factor for exponential resampling.

    axis : int
        The axis along which to transform `y`

    Returns
    -------
    x_scale : np.ndarray [dtype=complex]
        The scale transform of `y` along the `axis` dimension.

    Raises
    ------
    ParameterError
        if `n_fmt < 2` or `t_min <= 0`
        or if `y` is not finite
        or if `y.shape[axis] < 3`.

    Notes
    -----
    This function caches at level 30.


    Examples
    --------
    >>> # Generate a signal and time-stretch it (with energy normalization)
    >>> scale = 1.25
    >>> freq = 3.0
    >>> x1 = np.linspace(0, 1, num=1024, endpoint=False)
    >>> x2 = np.linspace(0, 1, num=scale * len(x1), endpoint=False)
    >>> y1 = np.sin(2 * np.pi * freq * x1)
    >>> y2 = np.sin(2 * np.pi * freq * x2) / np.sqrt(scale)
    >>> # Verify that the two signals have the same energy
    >>> np.sum(np.abs(y1)**2), np.sum(np.abs(y2)**2)
        (255.99999999999997, 255.99999999999969)
    >>> scale1 = librosa.fmt(y1, n_fmt=512)
    >>> scale2 = librosa.fmt(y2, n_fmt=512)
    >>> # And plot the results
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(8, 4))
    >>> plt.subplot(1, 2, 1)
    >>> plt.plot(y1, label='Original')
    >>> plt.plot(y2, linestyle='--', label='Stretched')
    >>> plt.xlabel('time (samples)')
    >>> plt.title('Input signals')
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')
    >>> plt.subplot(1, 2, 2)
    >>> plt.semilogy(np.abs(scale1), label='Original')
    >>> plt.semilogy(np.abs(scale2), linestyle='--', label='Stretched')
    >>> plt.xlabel('scale coefficients')
    >>> plt.title('Scale transform magnitude')
    >>> plt.legend(frameon=True)
    >>> plt.axis('tight')
    >>> plt.tight_layout()

    >>> # Plot the scale transform of an onset strength autocorrelation
    >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    ...                      offset=10.0, duration=30.0)
    >>> odf = librosa.onset.onset_strength(y=y, sr=sr)
    >>> # Auto-correlate with up to 10 seconds lag
    >>> odf_ac = librosa.autocorrelate(odf, max_size=10 * sr // 512)
    >>> # Normalize
    >>> odf_ac = librosa.util.normalize(odf_ac, norm=np.inf)
    >>> # Compute the scale transform
    >>> odf_ac_scale = librosa.fmt(librosa.util.normalize(odf_ac), n_fmt=512)
    >>> # Plot the results
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> plt.plot(odf, label='Onset strength')
    >>> plt.axis('tight')
    >>> plt.xlabel('Time (frames)')
    >>> plt.xticks([])
    >>> plt.legend(frameon=True)
    >>> plt.subplot(3, 1, 2)
    >>> plt.plot(odf_ac, label='Onset autocorrelation')
    >>> plt.axis('tight')
    >>> plt.xlabel('Lag (frames)')
    >>> plt.xticks([])
    >>> plt.legend(frameon=True)
    >>> plt.subplot(3, 1, 3)
    >>> plt.semilogy(np.abs(odf_ac_scale), label='Scale transform magnitude')
    >>> plt.axis('tight')
    >>> plt.xlabel('scale coefficients')
    >>> plt.legend(frameon=True)
    >>> plt.tight_layout()
    """

    n = y.shape[axis]

    if n < 3:
        raise ParameterError('y.shape[{:}]=={:} < 3'.format(axis, n))

    if t_min <= 0:
        raise ParameterError('t_min must be a positive number')

    if n_fmt is None:
        if over_sample < 1:
            raise ParameterError('over_sample must be >= 1')

        # The base is the maximum ratio between adjacent samples
        # Since the sample spacing is increasing, this is simply the
        # ratio between the positions of the last two samples: (n-1)/(n-2)
        log_base = np.log(n - 1) - np.log(n - 2)

        n_fmt = int(np.ceil(over_sample * (np.log(n - 1) - np.log(t_min)) / log_base))

    elif n_fmt < 3:
        raise ParameterError('n_fmt=={:} < 3'.format(n_fmt))
    else:
        log_base = (np.log(n_fmt - 1) - np.log(n_fmt - 2)) / over_sample

    if not np.all(np.isfinite(y)):
        raise ParameterError('y must be finite everywhere')

    base = np.exp(log_base)
    # original grid: signal covers [0, 1).  This range is arbitrary, but convenient.
    # The final sample is positioned at (n-1)/n, so we omit the endpoint
    x = np.linspace(0, 1, num=n, endpoint=False)

    # build the interpolator
    f_interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=axis)

    # build the new sampling grid
    # exponentially spaced between t_min/n and 1 (exclusive)
    # we'll go one past where we need, and drop the last sample
    # When over-sampling, the last input sample contributions n_over samples.
    # To keep the spacing consistent, we over-sample by n_over, and then
    # trim the final samples.
    n_over = int(np.ceil(over_sample))
    x_exp = np.logspace((np.log(t_min) - np.log(n)) / log_base,
                        0,
                        num=n_fmt + n_over,
                        endpoint=False,
                        base=base)[:-n_over]

    # Clean up any rounding errors at the boundaries of the interpolation
    # The interpolator gets angry if we try to extrapolate, so clipping is necessary here.
    if x_exp[0] < t_min or x_exp[-1] > float(n - 1.0) / n:
        x_exp = np.clip(x_exp, float(t_min) / n, x[-1])

    # Make sure that all sample points are unique
    assert len(np.unique(x_exp)) == len(x_exp)

    # Resample the signal
    y_res = f_interp(x_exp)

    # Broadcast the window correctly
    shape = [1] * y_res.ndim
    shape[axis] = -1

    # Apply the window and fft
    result = fft.fft(y_res * (x_exp**beta).reshape(shape),
                     axis=axis, overwrite_x=True)

    # Slice out the positive-scale component
    idx = [slice(None)] * result.ndim
    idx[axis] = slice(0, 1 + n_fmt//2)

    # Truncate and length-normalize
    return result[idx] * np.sqrt(n) / n_fmt


@cache(level=10)
def dct(n_filters, n_input):
    """Discrete cosine transform (DCT type-III) basis.

    .. [1] http://en.wikipedia.org/wiki/Discrete_cosine_transform

    Parameters
    ----------
    n_filters : int > 0 [scalar]
        number of output components (DCT filters)

    n_input : int > 0 [scalar]
        number of input components (frequency bins)

    Returns
    -------
    dct_basis: np.ndarray [shape=(n_filters, n_input)]
        DCT (type-III) basis vectors [1]_

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> n_fft = 2048
    >>> dct_filters = librosa.filters.dct(13, 1 + n_fft // 2)
    >>> dct_filters
    array([[ 0.031,  0.031, ...,  0.031,  0.031],
           [ 0.044,  0.044, ..., -0.044, -0.044],
           ...,
           [ 0.044,  0.044, ..., -0.044, -0.044],
           [ 0.044,  0.044, ...,  0.044,  0.044]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(dct_filters, x_axis='linear')
    >>> plt.ylabel('DCT function')
    >>> plt.title('DCT filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    basis = np.empty((n_filters, n_input))
    basis[0, :] = 1.0 / np.sqrt(n_input)

    samples = np.arange(1, 2*n_input, 2) * np.pi / (2.0 * n_input)

    for i in range(1, n_filters):
        basis[i, :] = np.cos(i*samples) * np.sqrt(2.0/n_input)

    return basis


@cache(level=10)
def mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False,
        norm=1):
    """Create a Filterbank matrix to combine FFT bins into Mel-frequency bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_mels    : int > 0 [scalar]
        number of Mel bands to generate

    fmin      : float >= 0 [scalar]
        lowest frequency (in Hz)

    fmax      : float >= 0 [scalar]
        highest frequency (in Hz).
        If `None`, use `fmax = sr / 2.0`

    htk       : bool [scalar]
        use HTK formula instead of Slaney

    norm : {None, 1, np.inf} [scalar]
        if 1, divide the triangular mel weights by the width of the mel band 
        (area normalization).  Otherwise, leave all the triangles aiming for 
        a peak value of 1.0

    Returns
    -------
    M         : np.ndarray [shape=(n_mels, 1 + n_fft/2)]
        Mel transform matrix

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    >>> melfb = librosa.filters.mel(22050, 2048)
    >>> melfb
    array([[ 0.   ,  0.016, ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           ...,
           [ 0.   ,  0.   , ...,  0.   ,  0.   ],
           [ 0.   ,  0.   , ...,  0.   ,  0.   ]])


    Clip the maximum frequency to 8KHz

    >>> librosa.filters.mel(22050, 2048, fmax=8000)
    array([[ 0.  ,  0.02, ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           ...,
           [ 0.  ,  0.  , ...,  0.  ,  0.  ],
           [ 0.  ,  0.  , ...,  0.  ,  0.  ]])


    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> librosa.display.specshow(melfb, x_axis='linear')
    >>> plt.ylabel('Mel filter')
    >>> plt.title('Mel filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    if fmax is None:
        fmax = float(sr) / 2

    if norm is not None and norm != 1 and norm != np.inf:
        raise ParameterError('Unsupported norm: {}'.format(repr(norm)))

    # Initialize the weights
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_fft // 2)))

    # Center freqs of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)

    # 'Center freqs' of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i+2] / fdiff[i+1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    if norm == 1:
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2:n_mels+2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

    # Only check weights if f_mel[0] is positive
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        # This means we have an empty channel somewhere
        warnings.warn('Empty filters detected in mel frequency basis. '
                      'Some channels will produce empty responses. '
                      'Try increasing your sampling rate (and fmax) or '
                      'reducing n_mels.')

    return weights


@cache(level=10)
def chroma(sr, n_fft, n_chroma=12, A440=440.0, ctroct=5.0,
           octwidth=2, norm=2, base_c=True):
    """Create a Filterbank matrix to convert STFT to chroma


    Parameters
    ----------
    sr        : number > 0 [scalar]
        audio sampling rate

    n_fft     : int > 0 [scalar]
        number of FFT bins

    n_chroma  : int > 0 [scalar]
        number of chroma bins

    A440      : float > 0 [scalar]
        Reference frequency for A440

    ctroct    : float > 0 [scalar]

    octwidth  : float > 0 or None [scalar]
        `ctroct` and `octwidth` specify a dominance window -
        a Gaussian weighting centered on `ctroct` (in octs, A0 = 27.5Hz)
        and with a gaussian half-width of `octwidth`.
        Set `octwidth` to `None` to use a flat weighting.

    norm : float > 0 or np.inf
        Normalization factor for each filter

    base_c : bool
        If True, the filter bank will start at 'C'.
        If False, the filter bank will start at 'A'.

    Returns
    -------
    wts : ndarray [shape=(n_chroma, 1 + n_fft / 2)]
        Chroma filter matrix

    See Also
    --------
    util.normalize
    feature.chroma_stft

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Build a simple chroma filter bank

    >>> chromafb = librosa.filters.chroma(22050, 4096)
    array([[  1.689e-05,   3.024e-04, ...,   4.639e-17,   5.327e-17],
           [  1.716e-05,   2.652e-04, ...,   2.674e-25,   3.176e-25],
    ...,
           [  1.578e-05,   3.619e-04, ...,   8.577e-06,   9.205e-06],
           [  1.643e-05,   3.355e-04, ...,   1.474e-10,   1.636e-10]])

    Use quarter-tones instead of semitones

    >>> librosa.filters.chroma(22050, 4096, n_chroma=24)
    array([[  1.194e-05,   2.138e-04, ...,   6.297e-64,   1.115e-63],
           [  1.206e-05,   2.009e-04, ...,   1.546e-79,   2.929e-79],
    ...,
           [  1.162e-05,   2.372e-04, ...,   6.417e-38,   9.923e-38],
           [  1.180e-05,   2.260e-04, ...,   4.697e-50,   7.772e-50]])


    Equally weight all octaves

    >>> librosa.filters.chroma(22050, 4096, octwidth=None)
    array([[  3.036e-01,   2.604e-01, ...,   2.445e-16,   2.809e-16],
           [  3.084e-01,   2.283e-01, ...,   1.409e-24,   1.675e-24],
    ...,
           [  2.836e-01,   3.116e-01, ...,   4.520e-05,   4.854e-05],
           [  2.953e-01,   2.888e-01, ...,   7.768e-10,   8.629e-10]])

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> specshow(chromafb, x_axis='linear')
    >>> plt.ylabel('Chroma filter')
    >>> plt.title('Chroma filter bank')
    >>> plt.colorbar()
    >>> plt.tight_layout()
    """

    wts = np.zeros((n_chroma, n_fft))

    # Get the FFT bins, not counting the DC component
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]

    frqbins = n_chroma * hz_to_octs(frequencies, A440)

    # make up a value for the 0 Hz bin = 1.5 octaves below bin 1
    # (so chroma is 50% rotated from bin 1, and bin width is broad)
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))

    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1],
                                              1.0), [1]))

    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype='d')).T

    n_chroma2 = np.round(float(n_chroma) / 2)

    # Project into range -n_chroma/2 .. n_chroma/2
    # add on fixed offset of 10*n_chroma to ensure all values passed to
    # rem are positive
    D = np.remainder(D + n_chroma2 + 10*n_chroma, n_chroma) - n_chroma2

    # Gaussian bumps - 2*D to make them narrower
    wts = np.exp(-0.5 * (2*D / np.tile(binwidthbins, (n_chroma, 1)))**2)

    # normalize each column
    wts = normalize(wts, norm=norm, axis=0)

    # Maybe apply scaling for fft bins
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((frqbins/n_chroma - ctroct)/octwidth)**2)),
            (n_chroma, 1))

    if base_c:
        wts = np.roll(wts, -3, axis=0)

    # remove aliasing columns, copy to ensure row-contiguity
    return np.ascontiguousarray(wts[:, :int(1 + n_fft/2)])


def __float_window(window_spec):
    '''Decorator function for windows with fractional input.

    This function guarantees that for fractional `x`, the following hold:

    1. `__float_window(window_function)(x)` has length `np.ceil(x)`
    2. all values from `np.floor(x)` are set to 0.

    For integer-valued `x`, there should be no change in behavior.
    '''

    def _wrap(n, *args, **kwargs):
        '''The wrapped window'''
        n_min, n_max = int(np.floor(n)), int(np.ceil(n))

        window = get_window(window_spec, n_min)

        if len(window) < n_max:
            window = np.pad(window, [(0, n_max - len(window))],
                            mode='constant')

        window[n_min:] = 0.0

        return window

    return _wrap


@cache(level=10)
def constant_q(sr, fmin=None, n_bins=84, bins_per_octave=12, tuning=0.0,
               window='hann', filter_scale=1, pad_fft=True, norm=1,
               **kwargs):
    r'''Construct a constant-Q basis.

    This uses the filter bank described by [1]_.

    .. [1] McVicar, Matthew.
            "A machine learning approach to automatic chord extraction."
            Dissertation, University of Bristol. 2013.


    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin. Defaults to `C1 ~= 32.70`

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin

    window : string, tuple, number, or function
        Windowing function to apply to filters.

    filter_scale : float > 0 [scalar]
        Scale of filter windows.
        Small values (<1) use shorter windows for higher temporal resolution.

    pad_fft : boolean
        Center-pad all filters up to the nearest integral power of 2.

        By default, padding is done with zeros, but this can be overridden
        by setting the `mode=` field in *kwargs*.

    norm : {inf, -inf, 0, float > 0}
        Type of norm to use for basis function normalization.
        See librosa.util.normalize

    kwargs : additional keyword arguments
        Arguments to `np.pad()` when `pad==True`.

    Returns
    -------
    filters : np.ndarray, `len(filters) == n_bins`
        `filters[i]` is `i`\ th time-domain CQT basis filter

    lengths : np.ndarray, `len(lengths) == n_bins`
        The (fractional) length of each filter

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    constant_q_lengths
    librosa.core.cqt
    librosa.util.normalize


    Examples
    --------
    Use a shorter window for each filter

    >>> basis, lengths = librosa.filters.constant_q(22050, filter_scale=0.5)

    Plot one octave of filters in time and frequency

    >>> import matplotlib.pyplot as plt
    >>> basis, lengths = librosa.filters.constant_q(22050)
    >>> plt.figure(figsize=(10, 6))
    >>> plt.subplot(2, 1, 1)
    >>> notes = librosa.midi_to_note(np.arange(24, 24 + len(basis)))
    >>> for i, (f, n) in enumerate(zip(basis, notes[:12])):
    ...     f_scale = librosa.util.normalize(f) / 2
    ...     plt.plot(i + f_scale.real)
    ...     plt.plot(i + f_scale.imag, linestyle=':')
    >>> plt.axis('tight')
    >>> plt.yticks(np.arange(len(notes[:12])), notes[:12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filters (one octave, time domain)')
    >>> plt.xlabel('Time (samples at 22050 Hz)')
    >>> plt.legend(['Real', 'Imaginary'], frameon=True, framealpha=0.8)
    >>> plt.subplot(2, 1, 2)
    >>> F = np.abs(np.fft.fftn(basis, axes=[-1]))
    >>> # Keep only the positive frequencies
    >>> F = F[:, :(1 + F.shape[1] // 2)]
    >>> librosa.display.specshow(F, x_axis='linear')
    >>> plt.yticks(np.arange(len(notes))[::12], notes[::12])
    >>> plt.ylabel('CQ filters')
    >>> plt.title('CQ filter magnitudes (frequency domain)')
    >>> plt.tight_layout()
    '''

    if fmin is None:
        fmin = note_to_hz('C1')

    # Pass-through parameters to get the filter lengths
    lengths = constant_q_lengths(sr, fmin,
                                 n_bins=n_bins,
                                 bins_per_octave=bins_per_octave,
                                 tuning=tuning,
                                 window=window,
                                 filter_scale=filter_scale)

    # Apply tuning correction
    correction = 2.0**(float(tuning) / bins_per_octave)
    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Convert lengths back to frequencies
    freqs = Q * sr / lengths

    # Build the filters
    filters = []
    for ilen, freq in zip(lengths, freqs):
        # Build the filter: note, length will be ceil(ilen)
        sig = np.exp(np.arange(ilen, dtype=float) * 1j * 2 * np.pi * freq / sr)

        # Apply the windowing function
        sig = sig * __float_window(window)(ilen)

        # Normalize
        sig = normalize(sig, norm=norm)

        filters.append(sig)

    # Pad and stack
    max_len = max(lengths)
    if pad_fft:
        max_len = int(2.0**(np.ceil(np.log2(max_len))))
    else:
        max_len = int(np.ceil(max_len))

    filters = np.asarray([pad_center(filt, max_len, **kwargs)
                          for filt in filters])

    return filters, np.asarray(lengths)


@cache(level=10)
def constant_q_lengths(sr, fmin, n_bins=84, bins_per_octave=12,
                       tuning=0.0, window='hann', filter_scale=1):
    r'''Return length of each filter in a constant-Q basis.

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    fmin : float > 0 [scalar]
        Minimum frequency bin.

    n_bins : int > 0 [scalar]
        Number of frequencies.  Defaults to 7 octaves (84 bins).

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float in `[-0.5, +0.5)` [scalar]
        Tuning deviation from A440 in fractions of a bin

    window : str or callable
        Window function to use on filters

    filter_scale : float > 0 [scalar]
        Resolution of filter windows. Larger values use longer windows.

    Returns
    -------
    lengths : np.ndarray
        The length of each filter.

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    constant_q
    librosa.core.cqt
    '''

    if fmin <= 0:
        raise ParameterError('fmin must be positive')

    if bins_per_octave <= 0:
        raise ParameterError('bins_per_octave must be positive')

    if filter_scale <= 0:
        raise ParameterError('filter_scale must be positive')

    if n_bins <= 0 or not isinstance(n_bins, int):
        raise ParameterError('n_bins must be a positive integer')

    correction = 2.0**(float(tuning) / bins_per_octave)

    fmin = correction * fmin

    # Q should be capitalized here, so we suppress the name warning
    # pylint: disable=invalid-name
    Q = float(filter_scale) / (2.0**(1. / bins_per_octave) - 1)

    # Compute the frequencies
    freq = fmin * (2.0 ** (np.arange(n_bins, dtype=float) / bins_per_octave))

    if freq[-1] * (1 + 0.5 * window_bandwidth(window) / Q) > sr / 2.0:
        raise ParameterError('Filter pass-band lies beyond Nyquist')

    # Convert frequencies to filter lengths
    lengths = Q * sr / freq

    return lengths


@cache(level=10)
def cq_to_chroma(n_input, bins_per_octave=12, n_chroma=12,
                 fmin=None, window=None, base_c=True):
    '''Convert a Constant-Q basis to Chroma.


    Parameters
    ----------
    n_input : int > 0 [scalar]
        Number of input components (CQT bins)

    bins_per_octave : int > 0 [scalar]
        How many bins per octave in the CQT

    n_chroma : int > 0 [scalar]
        Number of output bins (per octave) in the chroma

    fmin : None or float > 0
        Center frequency of the first constant-Q channel.
        Default: 'C1' ~= 32.7 Hz

    window : None or np.ndarray
        If provided, the cq_to_chroma filter bank will be
        convolved with `window`.

    base_c : bool
        If True, the first chroma bin will start at 'C'
        If False, the first chroma bin will start at 'A'

    Returns
    -------
    cq_to_chroma : np.ndarray [shape=(n_chroma, n_input)]
        Transformation matrix: `Chroma = np.dot(cq_to_chroma, CQT)`

    Raises
    ------
    ParameterError
        If `n_input` is not an integer multiple of `n_chroma`

    Notes
    -----
    This function caches at level 10.

    Examples
    --------
    Get a CQT, and wrap bins to chroma

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> CQT = librosa.cqt(y, sr=sr)
    >>> chroma_map = librosa.filters.cq_to_chroma(CQT.shape[0])
    >>> chromagram = chroma_map.dot(CQT)
    >>> # Max-normalize each time step
    >>> chromagram = librosa.util.normalize(chromagram, axis=0)

    >>> import matplotlib.pyplot as plt
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(CQT,
    ...                                                  ref=np.max),
    ...                          y_axis='cqt_note')
    >>> plt.title('CQT Power')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(chromagram, y_axis='chroma')
    >>> plt.title('Chroma (wrapped CQT)')
    >>> plt.colorbar()
    >>> plt.subplot(3, 1, 3)
    >>> chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    >>> librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    >>> plt.title('librosa.feature.chroma_stft')
    >>> plt.colorbar()
    >>> plt.tight_layout()

    '''

    # How many fractional bins are we merging?
    n_merge = float(bins_per_octave) / n_chroma

    if fmin is None:
        fmin = note_to_hz('C1')

    if np.mod(n_merge, 1) != 0:
        raise ParameterError('Incompatible CQ merge: '
                             'input bins must be an '
                             'integer multiple of output bins.')

    # Tile the identity to merge fractional bins
    cq_to_ch = np.repeat(np.eye(n_chroma), n_merge, axis=1)

    # Roll it left to center on the target bin
    cq_to_ch = np.roll(cq_to_ch, - int(n_merge // 2), axis=1)

    # How many octaves are we repeating?
    n_octaves = np.ceil(np.float(n_input) / bins_per_octave)

    # Repeat and trim
    cq_to_ch = np.tile(cq_to_ch, int(n_octaves))[:, :n_input]

    # What's the note number of the first bin in the CQT?
    # midi uses 12 bins per octave here
    midi_0 = np.mod(hz_to_midi(fmin), 12)

    if base_c:
        # rotate to C
        roll = midi_0
    else:
        # rotate to A
        roll = midi_0 - 9

    # Adjust the roll in terms of how many chroma we want out
    # We need to be careful with rounding here
    roll = int(np.round(roll * (n_chroma / 12.)))

    # Apply the roll
    cq_to_ch = np.roll(cq_to_ch, roll, axis=0).astype(float)

    if window is not None:
        cq_to_ch = scipy.signal.convolve(cq_to_ch,
                                         np.atleast_2d(window),
                                         mode='same')

    return cq_to_ch


def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    '''Uses a flexible heuristic to pick peaks in a signal.

    A sample n is selected as an peak if the corresponding x[n]
    fulfills the following three conditions:

    1. `x[n] == max(x[n - pre_max:n + post_max])`
    2. `x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta`
    3. `n - previous_n > wait`

    where `previous_n` is the last sample picked as a peak (greedily).

    This implementation is based on [1]_ and [2]_.

    .. [1] Boeck, Sebastian, Florian Krebs, and Markus Schedl.
        "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR.
        2012.

    .. [2] https://github.com/CPJKU/onset_detection/blob/master/onset_program.py


    Parameters
    ----------
    x         : np.ndarray [shape=(n,)]
        input signal to peak picks from

    pre_max   : int >= 0 [scalar]
        number of samples before `n` over which max is computed

    post_max  : int >= 1 [scalar]
        number of samples after `n` over which max is computed

    pre_avg   : int >= 0 [scalar]
        number of samples before `n` over which mean is computed

    post_avg  : int >= 1 [scalar]
        number of samples after `n` over which mean is computed

    delta     : float >= 0 [scalar]
        threshold offset for mean

    wait      : int >= 0 [scalar]
        number of samples to wait after picking a peak

    Returns
    -------
    peaks     : np.ndarray [shape=(n_peaks,), dtype=int]
        indices of peaks in `x`

    Raises
    ------
    ParameterError
        If any input lies outside its defined range

    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          hop_length=512,
    ...                                          aggregate=np.median)
    >>> peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
    >>> peaks
    array([  4,  23,  73, 102, 142, 162, 182, 211, 261, 301, 320,
           331, 348, 368, 382, 396, 411, 431, 446, 461, 476, 491,
           510, 525, 536, 555, 570, 590, 609, 625, 639])

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.frames_to_time(np.arange(len(onset_env)),
    ...                                sr=sr, hop_length=512)
    >>> plt.figure()
    >>> ax = plt.subplot(2, 1, 2)
    >>> D = librosa.stft(y)
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time')
    >>> plt.subplot(2, 1, 1, sharex=ax)
    >>> plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    >>> plt.vlines(times[peaks], 0,
    ...            onset_env.max(), color='r', alpha=0.8,
    ...            label='Selected peaks')
    >>> plt.legend(frameon=True, framealpha=0.8)
    >>> plt.axis('tight')
    >>> plt.tight_layout()
    '''

    if pre_max < 0:
        raise ParameterError('pre_max must be non-negative')
    if pre_avg < 0:
        raise ParameterError('pre_avg must be non-negative')
    if delta < 0:
        raise ParameterError('delta must be non-negative')
    if wait < 0:
        raise ParameterError('wait must be non-negative')

    if post_max <= 0:
        raise ParameterError('post_max must be positive')

    if post_avg <= 0:
        raise ParameterError('post_avg must be positive')

    if x.ndim != 1:
        raise ParameterError('input array must be one-dimensional')

    # Ensure valid index types
    pre_max = valid_int(pre_max, cast=np.ceil)
    post_max = valid_int(post_max, cast=np.ceil)
    pre_avg = valid_int(pre_avg, cast=np.ceil)
    post_avg = valid_int(post_avg, cast=np.ceil)
    wait = valid_int(wait, cast=np.ceil)

    # Get the maximum of the signal over a sliding window
    max_length = pre_max + post_max
    max_origin = np.ceil(0.5 * (pre_max - post_max))
    # Using mode='constant' and cval=x.min() effectively truncates
    # the sliding window at the boundaries
    mov_max = scipy.ndimage.filters.maximum_filter1d(x, int(max_length),
                                                     mode='constant',
                                                     origin=int(max_origin),
                                                     cval=x.min())

    # Get the mean of the signal over a sliding window
    avg_length = pre_avg + post_avg
    avg_origin = np.ceil(0.5 * (pre_avg - post_avg))
    # Here, there is no mode which results in the behavior we want,
    # so we'll correct below.
    mov_avg = scipy.ndimage.filters.uniform_filter1d(x, int(avg_length),
                                                     mode='nearest',
                                                     origin=int(avg_origin))

    # Correct sliding average at the beginning
    n = 0
    # Only need to correct in the range where the window needs to be truncated
    while n - pre_avg < 0 and n < x.shape[0]:
        # This just explicitly does mean(x[n - pre_avg:n + post_avg])
        # with truncation
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start:n + post_avg])
        n += 1
    # Correct sliding average at the end
    n = x.shape[0] - post_avg
    # When post_avg > x.shape[0] (weird case), reset to 0
    n = n if n > 0 else 0
    while n < x.shape[0]:
        start = n - pre_avg
        start = start if start > 0 else 0
        mov_avg[n] = np.mean(x[start:n + post_avg])
        n += 1

    # First mask out all entries not equal to the local max
    detections = x * (x == mov_max)

    # Then mask out all entries less than the thresholded average
    detections = detections * (detections >= (mov_avg + delta))

    # Initialize peaks array, to be filled greedily
    peaks = []

    # Remove onsets which are close together in time
    last_onset = -np.inf

    for i in np.nonzero(detections)[0]:
        # Only report an onset if the "wait" samples was reported
        if i > last_onset + wait:
            peaks.append(i)
            # Save last reported onset
            last_onset = i

    return np.array(peaks)
