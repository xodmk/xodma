# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodmaAudioTools.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
#
# XODMK Music Analysis Toolkit
#
# Cleansed Folded & Manipulated version of Librosa
#
# Read/Write .wav files
# import audio file as floating point data
# write audio data to text
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************


import os, sys
import six
import soundfile as sf
import numpy as np
import scipy.signal
import scipy.fftpack as fft
import resampy

currentDir = os.getcwd()
rootDir = os.path.dirname(currentDir)

sys.path.insert(0, rootDir + '/xodma/')

from xodmaMiscUtil import fix_length, valid_int
from xodmaParameterError import ParameterError

# import cache
from cache import cache

# // *---------------------------------------------------------------------* //

# temp python debugger - use >>>pdb.set_trace() to set break
# import pdb


# // *---------------------------------------------------------------------* //

__all__ = ['load_wav', 'write_wav', 'time_to_samples', 'samples_to_time',
           'buf_to_float', 'to_mono', 'resample', 'get_duration', 'autocorrelate',
           'zero_crossings', 'valid_audio',
           'peak_pick', 'xodmaPeaks', 'sig2txt']

# Resampling bandwidths as percentage of Nyquist
BW_BEST = resampy.filters.get_filter('kaiser_best')[2]
BW_FASTEST = resampy.filters.get_filter('kaiser_fast')[2]


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


def time_to_samples(times, sr=48000):
    """ Convert timestamps (in seconds) to sample indices.

    Parameters
    ----------
    times : np.ndarray
        Array of time values (in seconds)

    sr : number > 0
        Sampling rate

    Returns
    -------
    samples : np.ndarray [shape=times.shape, dtype=int]
        Sample indices corresponding to values in `times`

    See Also
    --------
    time_to_frames : convert time values to frame indices
    samples_to_time : convert sample indices to time values

    Examples
    --------
    # >> librosa.time_to_samples(np.arange(0, 1, 0.1), sr=22050)
    array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435,
           17640, 19845])

    """

    return (np.atleast_1d(times) * sr).astype(int)


def samples_to_time(samples, sr=48000):
    """ Convert sample indices to time (in seconds).

    Parameters
    ----------
    samples : np.ndarray
        Array of sample indices

    sr : number > 0
        Sampling rate

    Returns
    -------
    times : np.ndarray [shape=samples.shape, dtype=int]
        Time values corresponding to `samples` (in seconds)

    See Also
    --------
    samples_to_frames : convert sample indices to frame indices
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get timestamps corresponding to every 512 samples

    # >> librosa.samples_to_time(np.arange(0, 22050, 512))
    array([ 0.   ,  0.023,  0.046,  0.07 ,  0.093,  0.116,  0.139,
            0.163,  0.186,  0.209,  0.232,  0.255,  0.279,  0.302,
            0.325,  0.348,  0.372,  0.395,  0.418,  0.441,  0.464,
            0.488,  0.511,  0.534,  0.557,  0.58 ,  0.604,  0.627,
            0.65 ,  0.673,  0.697,  0.72 ,  0.743,  0.766,  0.789,
            0.813,  0.836,  0.859,  0.882,  0.906,  0.929,  0.952,
            0.975,  0.998])
    """

    return np.atleast_1d(samples) / float(sr)


def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """ Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    See Also
    --------
    buf_to_float

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in `x`

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1. / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


@cache(level=20)
def to_mono(y):
    """ Force an audio signal down to mono.

    Parameters
    ----------
    y : np.ndarray [shape=(2,n) or shape=(n,)]
        audio time series, either stereo or mono

    Returns
    -------
    y_mono : np.ndarray [shape=(n,)]
        `y` as a monophonic time-series

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    # >> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
    # >> y.shape
    (2, 1355168)
    # >> y_mono = librosa.to_mono(y)
    # >> y_mono.shape
    (1355168,)

    """

    # Validate the buffer.  Stereo is ok here.
    valid_audio(y, mono=False)

    if y.ndim > 1:
        y = np.mean(y, axis=0)

    return y


@cache(level=20)
def resample(y, orig_sr, target_sr, res_type='kaiser_best', fix=True, scale=False, **kwargs):
    """ Resample a time series from orig_sr to target_sr

    Parameters
    ----------
    y : np.ndarray [shape=(n,) or shape=(2, n)]
        audio time series.  Can be mono or stereo.

    orig_sr : number > 0 [scalar]
        original sampling rate of `y`

    target_sr : number > 0 [scalar]
        target sampling rate

    res_type : str
        resample type (see note)

        .. note::
            By default, this uses `resampy`'s high-quality mode ('kaiser_best').

            To use a faster method, set `res_type='kaiser_fast'`.

            To use `scipy.signal.resample`, set `res_type='scipy'`.

    fix : bool
        adjust the length of the resampled signal to be of size exactly
        `ceil(target_sr * len(y) / orig_sr)`

    scale : bool
        Scale the resampled signal so that `y` and `y_hat` have approximately
        equal total energy.

    kwargs : additional keyword arguments
        If `fix==True`, additional keyword arguments to pass to
        `librosa.util.fix_length`.

    Returns
    -------
    y_hat : np.ndarray [shape=(n * target_sr / orig_sr,)]
        `y` resampled from `orig_sr` to `target_sr`


    See Also
    --------
    librosa.util.fix_length
    scipy.signal.resample
    resampy.resample

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Downsample from 22 KHz to 8 KHz

    # >> y, sr = librosa.load(librosa.util.example_audio_file(), sr=22050)
    # >> y_8k = librosa.resample(y, sr, 8000)
    # >> y.shape, y_8k.shape
    ((1355168,), (491671,))
    """

    # First, validate the audio buffer
    valid_audio(y, mono=False)

    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr

    n_samples = int(np.ceil(y.shape[-1] * ratio))

    if res_type == 'scipy':
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)

    return np.ascontiguousarray(y_hat, dtype=y.dtype)


def get_duration(y=None, sr=48000, S=None, n_fft=2048, hop_length=512,
                 center=True, filename=None):
    """ Compute the duration (in seconds) of an audio time series,
    feature matrix, or filename.

    Examples
    --------
    # >> # Load the example audio file
    # >> y, sr = librosa.load(librosa.util.example_audio_file())
    # >> librosa.get_duration(y=y, sr=sr)
    61.44

    # >> # Or directly from an audio file
    # >> librosa.get_duration(filename=librosa.util.example_audio_file())
    61.4

    # >> # Or compute duration from an STFT matrix
    # >> y, sr = librosa.load(librosa.util.example_audio_file())
    # >> S = librosa.stft(y)
    # >> librosa.get_duration(S=S, sr=sr)
    61.44

    # >> # Or a non-centered STFT matrix
    # >> S_left = librosa.stft(y, center=False)
    # >> librosa.get_duration(S=S_left, sr=sr)
    61.3471201814059

    Parameters
    ----------
    y : np.ndarray [shape=(n,), (2, n)] or None
        audio time series

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    S : np.ndarray [shape=(d, t)] or None
        STFT matrix, or any STFT-derived matrix (e.g., chromagram
        or mel spectrogram).

    n_fft       : int > 0 [scalar]
        FFT window size for `S`

    hop_length  : int > 0 [ scalar]
        number of audio samples between columns of `S`

    center  : boolean
        - If `True`, `S[:, t]` is centered at `y[t * hop_length]`
        - If `False`, then `S[:, t]` begins at `y[t * hop_length]`

    filename : str
        If provided, all other parameters are ignored, and the
        duration is calculated directly from the audio file.
        Note that this avoids loading the contents into memory,
        and is therefore useful for querying the duration of
        long files.

    Returns
    -------
    d : float >= 0
        Duration (in seconds) of the input time series or spectrogram.
    """

    if filename is not None:
        with sf.read(filename) as fdesc:
            return fdesc.duration

    if y is None:
        assert S is not None

        n_frames = S.shape[1]
        n_samples = n_fft + hop_length * (n_frames - 1)

        # If centered, we lose half a window from each end of S
        if center:
            n_samples = n_samples - 2 * int(n_fft / 2)

    else:
        # Validate the audio buffer.  Stereo is okay here.
        valid_audio(y, mono=False)
        if y.ndim == 1:
            n_samples = len(y)
        else:
            n_samples = y.shape[-1]

    return float(n_samples) / sr


@cache(level=20)
def autocorrelate(y, max_size=None, axis=-1):
    """ Bounded auto-correlation

    Parameters
    ----------
    y : np.ndarray
        array to autocorrelate

    max_size  : int > 0 or None
        maximum correlation lag.
        If unspecified, defaults to `y.shape[axis]` (unbounded)

    axis : int
        The axis along which to autocorrelate.
        By default, the last axis (-1) is taken.

    Returns
    -------
    z : np.ndarray
        truncated autocorrelation `y*y` along the specified axis.
        If `max_size` is specified, then `z.shape[axis]` is bounded
        to `max_size`.

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    Compute full autocorrelation of y

    # >> y, sr = librosa.load(librosa.util.example_audio_file(), offset=20, duration=10)
    # >> librosa.autocorrelate(y)
    array([  3.226e+03,   3.217e+03, ...,   8.277e-04,   3.575e-04], dtype=float32)

    Compute onset strength auto-correlation up to 4 seconds

    # >> import matplotlib.pyplot as plt
    # >> odf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=512)
    # >> ac = librosa.autocorrelate(odf, max_size=4* sr / 512)
    # >> plt.plot(ac)
    # >> plt.title('Auto-correlation')
    # >> plt.xlabel('Lag (frames)')

    """

    if max_size is None:
        max_size = y.shape[axis]

    max_size = int(min(max_size, y.shape[axis]))

    # Compute the power spectrum along the chosen axis
    # Pad out the signal to support full-length auto-correlation.
    powspec = np.abs(fft.fft(y, n=2 * y.shape[axis] + 1, axis=axis)) ** 2

    # Convert back to time domain
    autocorr = fft.ifft(powspec, axis=axis, overwrite_x=True)

    # Slice down to max_size
    subslice = [slice(None)] * autocorr.ndim
    subslice[axis] = slice(max_size)

    autocorr = autocorr[subslice]

    if not np.iscomplexobj(y):
        autocorr = autocorr.real

    return autocorr


@cache(level=20)
def zero_crossings(y, threshold=1e-10, ref_magnitude=None, pad=True,
                   zero_pos=True, axis=-1):
    """ Find the zero-crossings of a signal `y`: indices `i` such that
    `sign(y[i]) != sign(y[j])`.

    If `y` is multi-dimensional, then zero-crossings are computed along
    the specified `axis`.


    Parameters
    ----------
    y : np.ndarray
        The input array

    threshold : float > 0 or None
        If specified, values where `-threshold <= y <= threshold` are
        clipped to 0.

    ref_magnitude : float > 0 or callable
        If numeric, the threshold is scaled relative to `ref_magnitude`.

        If callable, the threshold is scaled relative to
        `ref_magnitude(np.abs(y))`.

    pad : boolean
        If `True`, then `y[0]` is considered a valid zero-crossing.

    zero_pos : boolean
        If `True` then the value 0 is interpreted as having positive sign.

        If `False`, then 0, -1, and +1 all have distinct signs.

    axis : int
        Axis along which to compute zero-crossings.

    Returns
    -------
    zero_crossings : np.ndarray [shape=y.shape, dtype=boolean]
        Indicator array of zero-crossings in `y` along the selected axis.

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    # >> # Generate a time-series
    # >> y = np.sin(np.linspace(0, 4 * 2 * np.pi, 20))
    # >> y
    array([  0.000e+00,   9.694e-01,   4.759e-01,  -7.357e-01,
            -8.372e-01,   3.247e-01,   9.966e-01,   1.646e-01,
            -9.158e-01,  -6.142e-01,   6.142e-01,   9.158e-01,
            -1.646e-01,  -9.966e-01,  -3.247e-01,   8.372e-01,
             7.357e-01,  -4.759e-01,  -9.694e-01,  -9.797e-16])
    # >> # Compute zero-crossings
    # >> z = librosa.zero_crossings(y)
    # >> z
    array([ True, False, False,  True, False,  True, False, False,
            True, False,  True, False,  True, False, False,  True,
           False,  True, False,  True], dtype=bool)
    # >> # Stack y against the zero-crossing indicator
    # >> np.vstack([y, z]).T
    array([[  0.000e+00,   1.000e+00],
           [  9.694e-01,   0.000e+00],
           [  4.759e-01,   0.000e+00],
           [ -7.357e-01,   1.000e+00],
           [ -8.372e-01,   0.000e+00],
           [  3.247e-01,   1.000e+00],
           [  9.966e-01,   0.000e+00],
           [  1.646e-01,   0.000e+00],
           [ -9.158e-01,   1.000e+00],
           [ -6.142e-01,   0.000e+00],
           [  6.142e-01,   1.000e+00],
           [  9.158e-01,   0.000e+00],
           [ -1.646e-01,   1.000e+00],
           [ -9.966e-01,   0.000e+00],
           [ -3.247e-01,   0.000e+00],
           [  8.372e-01,   1.000e+00],
           [  7.357e-01,   0.000e+00],
           [ -4.759e-01,   1.000e+00],
           [ -9.694e-01,   0.000e+00],
           [ -9.797e-16,   1.000e+00]])
    # >> # Find the indices of zero-crossings
    # >> np.nonzero(z)
    (array([ 0,  3,  5,  8, 10, 12, 15, 17, 19]),)
    """

    # Clip within the threshold
    if threshold is None:
        threshold = 0.0

    if six.callable(ref_magnitude):
        threshold = threshold * ref_magnitude(np.abs(y))

    elif ref_magnitude is not None:
        threshold = threshold * ref_magnitude

    if threshold > 0:
        y = y.copy()
        y[np.abs(y) <= threshold] = 0

    # Extract the sign bit
    if zero_pos:
        y_sign = np.signbit(y)
    else:
        y_sign = np.sign(y)

    # Find the change-points by slicing
    slice_pre = [slice(None)] * y.ndim
    slice_pre[axis] = slice(1, None)

    slice_post = [slice(None)] * y.ndim
    slice_post[axis] = slice(-1)

    # Since we've offset the input by one, pad back onto the front
    padding = [(0, 0)] * y.ndim
    padding[axis] = (1, 0)

    return np.pad((y_sign[slice_post] != y_sign[slice_pre]),
                  padding,
                  mode='constant',
                  constant_values=pad)


def valid_audio(y, mono=True):
    """ Validate whether a variable contains valid, mono audio data.


    Parameters
    ----------
    y : np.ndarray
      The input data to validate

    mono : bool
      Whether or not to force monophonic audio

    Returns
    -------
    valid : bool
        True if all tests pass

    Raises
    ------
    ParameterError
        If `y` fails to meet the following criteria:
            - `type(y)` is `np.ndarray`
            - `mono == True` and `y.ndim` is not 1
            - `mono == False` and `y.ndim` is not 1 or 2
            - `np.isfinite(y).all()` is not True

    Notes
    -----
    This function caches at level 20.

    Examples
    --------
    # >> # Only allow monophonic signals
    # >> y, sr = librosa.load(librosa.util.example_audio_file())
    # >> librosa.util.valid_audio(y)
    True

    # >> # If we want to allow stereo signals
    # >> y, sr = librosa.load(librosa.util.example_audio_file(), mono=False)
    # >> librosa.util.valid_audio(y, mono=False)
    True
    """

    if not isinstance(y, np.ndarray):
        print('ERROR: data must be of type numpy.ndarray')
        return False

    if mono and y.ndim != 1:
        print('ERROR: Invalid shape for monophonic audio: '
              'ndim={:d}, shape={}'.format(y.ndim, y.shape))
        return False

    elif y.ndim > 2:
        print('ERROR: Invalid shape for audio: '
              'ndim={:d}, shape={}'.format(y.ndim, y.shape))
        return False

    if not np.isfinite(y).all():
        print('ERROR: Audio buffer is not finite everywhere')
        return False

    return True


# // *---------------------------------------------------------------------* //
# // *--.WAV Read / Write func --*
# // *---------------------------------------------------------------------* //


# inputs:  wavIn, audioSrcDir, wavLength
# outputs: ySrc_ch1, ySrc_ch2, numChannels, fs, ySamples


# Load Stereo/mono .wav file

def load_wav(wavInPath, wavLength, printInfo=False):
    """ Load an audio file as a floating point time series.
        wavInPath: path to .wav file
        wavLength: length of audio to load in seconds (0 = full length)
        info: prints details of loaded .wav to screen

        returns: [xSrc, numChannels, fs, xSamples]
        to unpack:
          if numChannels==1:
            xSrc_ch1 = xSrc

          elif numChannels==2:
            xSrc_ch1 = xSrc[:,0]
            xSrc_ch2 = xSrc[:,1] """

    audioSrc = wavInPath

    # with open(audioSrc, 'rb') as f:
    #    ySrc, ySrcSr = sf.read(f)

    # ySrc, ySrcSr = sf.read(audioSrc, channels=1, samplerate=44100, subtype='FLOAT')

    numChannels = sf.info(audioSrc).channels
    # STEREO or MONO SOURCE WAVE
    # ** wavLength==0 - use full length of src .wav file **
    if (wavLength == 0):
        xSrc, fs = sf.read(audioSrc)
        xSamples = len(xSrc)
        xLength = samples_to_time(xSamples, fs)[0]
    else:
        xLength = wavLength
        fsTmp = sf.info(audioSrc).samplerate
        durTmp = sf.info(audioSrc).duration
        if xLength > durTmp:
            sys.exit('ERROR: wavLength setting exceeds the length of audio source')
        xSamples = int(time_to_samples(xLength, fsTmp))

        if numChannels == 1:
            xSrc, fs = sf.read(audioSrc, channels=1, frames=xSamples)
        elif numChannels == 2:
            xSrc, fs = sf.read(audioSrc, frames=xSamples)

    #    if numChannels==1:
    #        xSrc_ch1 = xSrc
    #        xSrc_ch2 = 0
    #    elif numChannels==2:
    #        xSrc_ch1 = xSrc[:,0]
    #        xSrc_ch2 = xSrc[:,1]

    numChannels = len(np.shape(xSrc))

    if printInfo == True:
        # length of input signal - '0' => length of input .wav file
        print('number of Channels = ' + str(len(np.shape(xSrc))))
        print('length of input signal in seconds: ----- ' + str(xLength))
        print('length of input signal in samples: ----- ' + str(xSamples))
        print('audio sample rate: --------------------- ' + str(fs) + '\n')

    return xSrc, numChannels, fs, xLength, xSamples


def write_wav(path, y, sr, norm=False):
    """ Output a time series as a .wav file

    Parameters
    ----------
    path : str
        path to save the output wav file

    y : np.ndarray [shape=(n,) or (2,n)]
        audio time series (mono or stereo)

    sr : int > 0 [scalar]
        sampling rate of `y`

    norm : boolean [scalar]
        enable amplitude normalization.
        For floating point `y`, scale the data to the range [-1, +1].

    Examples
    --------
    Trim a signal to 5 seconds and save it back

    # >> y, sr = audioTools.load(earSrc, sr=None, duration=5.0)
    ...
    # >> audioTools.write_wav('myNewWave_5s.wav', y, sr)

    """

    # Validate the buffer.  Stereo is okay here.
    valid_audio(y, mono=False)

    wav = y

    # Check for stereo
    if wav.ndim > 1 and wav.shape[0] == 2:
        wav = wav.T

    # Save
    sf.write(path, wav, sr)

# // *---------------------------------------------------------------------* //


def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
    """ Uses a flexible heuristic to pick peaks in a signal.
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
        threshold offset above windowed-mean to qualify a peak
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
    # >>> y, sr = librosa.load(librosa.util.example_audio_file(), duration=15)
    # >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    # ...                                          hop_length=512,
    # ...                                          aggregate=np.median)
    # >>> peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
    # >>> peaks
    # array([  4,  23,  73, 102, 142, 162, 182, 211, 261, 301, 320,
    #        331, 348, 368, 382, 396, 411, 431, 446, 461, 476, 491,
    #        510, 525, 536, 555, 570, 590, 609, 625, 639])
    #
    # >>> import matplotlib.pyplot as plt
    # >>> times = librosa.frames_to_time(np.arange(len(onset_env)),
    # ...                                sr=sr, hop_length=512)
    # >>> plt.figure()
    # >>> ax = plt.subplot(2, 1, 2)
    # >>> D = librosa.stft(y)
    # >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    # ...                          y_axis='log', x_axis='time')
    # >>> plt.subplot(2, 1, 1, sharex=ax)
    # >>> plt.plot(times, onset_env, alpha=0.8, label='Onset strength')
    # >>> plt.vlines(times[peaks], 0,
    # ...            onset_env.max(), color='r', alpha=0.8,
    # ...            label='Selected peaks')
    # >>> plt.legend(frameon=True, framealpha=0.8)
    # >>> plt.axis('tight')
    # >>> plt.tight_layout()
    """

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


def xodmaPeaks(sigIn, sr, hop, peakThresh, peakWait, **kwargs):
    """
    Wrapper for XODMA peak_pick function (modified clone of librosa func)
    sigIn      : input 1D time domain signal (audio, signal, envelope, etc.)
    sr         : sample rate (normalized for STFT 48KHz audio)
    hop        : STFT Hop length (normalized for hop length 256 (?!))
    peakThresh : threshold offset above windowed-mean to qualify a peak
    peakWait   : number of samples to wait after picking a peak

    ** FIXIT FIXIT - good results, but nonsensical parameterization...
    Current Example:
    # Matches: peaks = peak_pick(onset_env, 7, 7, 7, 7, 0.5, 5)  # -> default librosa params
    # >> peakThresh = 0.5
    # >> peakWait = 5
    """

    kwargs.setdefault('pre_max', 0.04 * sr // hop)  # 7.0
    kwargs.setdefault('post_max', 0.04 * sr // hop)  # 7.0
    kwargs.setdefault('pre_avg', 0.04 * sr // hop)  # 7.0
    kwargs.setdefault('post_avg', 0.04 * sr // hop)  # 7.0
    kwargs.setdefault('delta', peakThresh)
    kwargs.setdefault('wait', peakWait)  # 30ms

    peaks = peak_pick(sigIn, **kwargs)

    return peaks


# // *---------------------------------------------------------------------* //
# begin : file output
# // *---------------------------------------------------------------------* //

# // *-----------------------------------------------------------------* //
# // *---TXT write simple periodic waveforms (sigLength # samples)
# // *-----------------------------------------------------------------* //

def sig2txt(sigIn, outNm, outDir='None', comma=False):
    """ writes data to TXT file
        signal output name = outNm (expects string)

        Example Usage:

        # rootDir = 'C:/odmkDev/odmkCode/odmkPython/'
        outputDir = rootDir+'DSP/werk/'

        # // *-----------------------------------------------
        # write a 1D sine signal to .txt
        sigIn = sin2_5K

        outNmTXT = 'sin2_5K_src.txt'
        nChan = 1

        sig2txt(sigIn, nChan, outNmTXT, outDir=outputDir)
        print('\nwrote data to file: '+outputDir+outNmTXT)


        # // *-----------------------------------------------
        # write a stereo wave file to .txt
        sigIn = amenBreak

        outNmTXT = 'amenBreak_src.txt'
        nChan = len(np.shape(amenBreak))

        sig2txt(sigIn, nChan, outNmTXT, outDir=outputDir)
        print('\nwrote data to file: '+outputDir+outNmTXT)

        # // *-----------------------------------------------
        # write multi-array of 1D sine signals to .txt
        sigIn = yOrthoScaleArray

        outNmTXT = 'yOrthoScaleArray.txt'
        nChan = len(yOrthoScaleArray)

        sig2txt(sigIn, nChan, outNmTXT, outDir=outputDir)
        print('\nwrote data to file: '+outputDir+outNmTXT)
    """

    if comma:
        try:
            if isinstance(comma, bool):
                insertCommas = True
        except NameError:
            print('Error: comma must be True or False')
    else:
        insertCommas = False

    if outDir != 'None':
        try:
            if isinstance(outDir, str):
                txtOutDir = outDir
                os.makedirs(txtOutDir, exist_ok=True)
        except NameError:
            print('Error: outNm must be a string')
    else:
        txtOutDir = './'

    try:
        if isinstance(outNm, str):
            sigOutFull = txtOutDir + outNm
    except NameError:
        print('Error: outNm must be a string')

    # writes data to .TXT file:
    outputFile = open(sigOutFull, 'w', newline='')

    # pdb.set_trace()

    nChan = sigIn.ndim

    if nChan == 0:
        print('ERROR: Number of Channels must be >= 1')
    elif nChan == 1:
        for i in range(len(sigIn)):
            if insertCommas:
                if i == (len(sigIn) - 1):
                    outputFile.write(str(sigIn[i]) + '\n')
                else:
                    outputFile.write(str(sigIn[i]) + ',\n')
            else:
                outputFile.write(str(sigIn[i]) + '\n')
    else:
        for i in range(len(sigIn[0])):
            lineTmp = ""
            if insertCommas:
                for j in range(len(sigIn) - 1):
                    strTmp = str(sigIn[j, i]) + str(',    ')
                    lineTmp = lineTmp + strTmp
                if i == (len(sigIn[0]) - 1):
                    lineTmp = lineTmp + str(sigIn[len(sigIn) - 1, i]) + '\n'
                else:
                    lineTmp = lineTmp + str(sigIn[len(sigIn) - 1, i]) + ',\n'
            else:
                for j in range(len(sigIn) - 1):
                    strTmp = str(sigIn[j, i]) + str('    ')
                    lineTmp = lineTmp + strTmp
                lineTmp = lineTmp + str(sigIn[len(sigIn) - 1, i]) + '\n'

            outputFile.write(lineTmp)

    outputFile.close()


# // *-----------------------------------------------------------------* //
# // *---CSV write simple periodic waveforms (sigLength # samples)
# // *-----------------------------------------------------------------* //

# def sig2csv(sigIn, outNm, outDir='None'):
#    ''' writes data to CSV file
#        signal output name = outNm (expects string) '''
#
#    if outDir != 'None':
#        try:
#            if isinstance(outDir, str):
#                csvOutDir = outDir
#                os.makedirs(csvOutDir, exist_ok=True)
#        except NameError:
#            print('Error: outNm must be a string')
#    else:
#        csvOutDir = './'
#
#    try:
#        if isinstance(outNm, str):
#            sigOutFull = csvOutDir+outNm
#    except NameError:
#        print('Error: outNm must be a string')
#
#    # writes data to .CSV file:
#    outputFile = open(sigOutFull, 'w', newline='')
#    outputWriter = csv.writer(outputFile)
#
#    for i in range(len(sigIn)):
#        tmpRow = [sigIn[i]]
#        outputWriter.writerow(tmpRow)
#
#    outputFile.close()
#    
#    # example usage : Write Signal to .csv File
#    #outNmCSV = 'sin2_5K_src.csv'
#
#    # rootDir = 'C:/odmkDev/odmkCode/odmkPython/'
#    # outputDir = rootDir+'DSP/werk/'
#    # sig2csv(sigIn, outNmCSV, outDir=outputDir)
#    
#    # // *-----------------------------------------------------------------* //


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# end : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

# // *---------------------------------------------------------------------* //
