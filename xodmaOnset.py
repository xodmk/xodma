# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

# __::((xodmaOnset.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
#
# XODMK Audio Tools - Onset Detection / Peak finding
#
#       
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt


currentDir = os.getcwd()
rootDir = os.path.dirname(currentDir)
sys.path.insert(0, rootDir+'/xodma')

from xodmaAudioTools import peak_pick
from xodmaSpectralUtil import frames_to_samples, frames_to_time
from xodmaMiscUtil import fix_frames, match_events, sync
from xodmaSpectralTools import power_to_db
from xodmaSpectralFeature import melspectrogram
from xodmaParameterError import ParameterError
from cache import cache

# removed from xodma -> moved to xodSpectral/xodOnset_tb.py
# from xodmaSpectralPlot import specshow
# from xodmaSpectralTools import magphase, amplitude_to_db, stft

#sys.path.insert(1, rootDir+'/xodUtil')
#import xodPlotUtil as xodplt

# temp python debugger - use >>>pdb.set_trace() to set break
import pdb


__all__ = ['detectOnset',
           'onset_detect',
           'onset_strength',
           'onset_strength_multi',
           'onset_backtrack',
           'get_peak_regions',
           'getOnsetSampleSegments',
           'getOnsetTimeSegments']


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

def detectOnset(y, peakThresh, peakWait, hop_length=512, sr=48000,
                backtrack=False, **kwargs):
    
    """Basic onset detector.  Locate note onset events by picking peaks in an
    onset strength envelope.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [1]_.

    .. [1] https://github.com/CPJKU/onset_db


    Parameters
    ----------
    y          : np.ndarray [shape=(n,)]
        audio time series
        
    peakThresh : controls threshold of onset detection
        (minimum 0.05 ~ 9.0(?))
    
    peakWait   : controls spacing of onset detections
        (minimum 0.03 ~ .wav length(?)) - long wait = fewer onsets

    sr         : number > 0 [scalar]
        sampling rate of `y`

    onset_envelope     : np.ndarray [shape=(m,)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    units : {'frames', 'samples', 'time'}
        The units to encode detected onset events in.
        By default, 'frames' are used.

    backtrack : bool
        If `True`, detected onset events are backtracked to the nearest
        preceding minimum of `energy`.

        This is primarily useful when using onsets as slice points for segmentation.

    energy : np.ndarray [shape=(m,)] (optional)
        An energy function to use for backtracking detected onset events.
        If none is provided, then `onset_envelope` is used.

    kwargs : placeholder for internal use (additional keyword arguments
        Additional parameters for peak picking.)

        See `librosa.util.peak_pick` for details.


    Returns
    -------

    onsets : np.ndarray [shape=(n_onsets,)]
        estimated positions of detected onsets, in whichever units
        are specified.  By default, frame indices.

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty list.


    Raises
    ------
    ParameterError
        if neither `y` nor `onsets` are provided

        or if `units` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    onset_strength : compute onset strength per-frame
    onset_backtrack : backtracking onset events
    librosa.util.peak_pick : pick peaks from a time series


    Examples
    --------
    Get onset times from a signal

    >> y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=2.0)
    >> onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    >> librosa.frames_to_time(onset_frames, sr=sr)
    array([ 0.07 ,  0.395,  0.511,  0.627,  0.766,  0.975,
            1.207,  1.324,  1.44 ,  1.788,  1.881])

    Or use a pre-computed onset envelope

    >> o_env = librosa.onset.onset_strength(y, sr=sr)
    >> times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    >> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    """

    onset_env = onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)

    # peak_pick
    # peaks = peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        
    #    pre_max   : int >= 0 [scalar]
    #        number of samples before `n` over which max is computed
    #
    #    post_max  : int >= 1 [scalar]
    #        number of samples after `n` over which max is computed
    #
    #    pre_avg   : int >= 0 [scalar]
    #        number of samples before `n` over which mean is computed
    #
    #    post_avg  : int >= 1 [scalar]
    #        number of samples after `n` over which mean is computed
    #
    #    delta     : float >= 0 [scalar]
    #        threshold offset for mean
    #
    #    wait      : int >= 0 [scalar]
    #        number of samples to wait after picking a peak
    #
    #    Returns
    #    -------
    #    peaks     : np.ndarray [shape=(n_peaks,), dtype=int]
    #        indices of peaks in `x`
    
    # peaks = peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
    # peaks = peak_pick(onset_env, 6, 6, 6, 6, 0.5, 8)
    # peaks = peak_pick(onset_env, 7, 7, 7, 7, 0.5, 7)
    # peaks = peak_pick(onset_env, 9, 9, 9, 9, 0.5, 7)
    # peaks = peak_pick(onset_env, 12, 12, 12, 12, 0.5, 6)
    # peaks = peak_pick(onset_env, 32, 32, 32, 32, 0.5, 32)
    # peaks = peak_pick(onset_env, 64, 64, 64, 64, 0.5, 64)

    # peaks = peak_pick(onset_env, pkctrl, pkctrl, pkctrl, pkctrl, 0.5, pkctrl)
    
    # peak_onsets_ch1 = np.array(onset_env_ch1)[peaks_ch1]
    # peak_onsets_ch2 = np.array(onset_env_ch2)[peaks_ch2]

    # These parameter settings found by large-scale search
    # kwargs.setdefault('pre_max', 0.03 * sr // hop_length)       # 30ms
    # kwargs.setdefault('post_max', 0.00 * sr // hop_length + 1)  # 0ms
    # kwargs.setdefault('pre_avg', 0.10 * sr // hop_length)       # 100ms
    # kwargs.setdefault('post_avg', 0.10 * sr // hop_length + 1)  # 100ms
    # kwargs.setdefault('wait', 0.03 * sr // hop_length)          # 30ms
    # kwargs.setdefault('delta', 0.07)

    kwargs.setdefault('pre_max', 0.03 * sr // hop_length)       # 30ms
    kwargs.setdefault('post_max', 0.00 * sr // hop_length + 1)  # 0ms
    kwargs.setdefault('pre_avg', 0.10 * sr // hop_length)       # 100ms
    kwargs.setdefault('post_avg', 0.10 * sr // hop_length + 1)  # 100ms
    kwargs.setdefault('wait', peakWait * sr // hop_length)      # 30ms
    kwargs.setdefault('delta', peakThresh)

    # Peak pick the onset envelope
    onsets = peak_pick(onset_env, **kwargs)

    # Optionally backtrack the events
    if backtrack:
        onsets = onset_backtrack(onsets, onset_env)

    onsets_samples = frames_to_samples(onsets, hop_length=hop_length)
    onsets_time = frames_to_time(onsets, hop_length=hop_length, sr=sr)

    # // *-----------------------------------------------------------------* //
    # // *--- Calculate Peak Regions (# frames of peak regions) ---*

    # peak_regions = get_peak_regions(peaks, len(onset_env))

    # # // *--- Plot - source signal ---*
    #
    # if plots > 1:
    #
    #     fnum = 3
    #     pltTitle = 'Input Signals: aSrc_ch1'
    #     pltXlabel = 'sinArray time-domain wav'
    #     pltYlabel = 'Magnitude'
    #
    #     # define a linear space from 0 to 1/2 Fs for x-axis:
    #     xaxis = np.linspace(0, len(y), len(y))
    #
    #     xodplt.xodPlot1D(fnum, y, xaxis, pltTitle, pltXlabel, pltYlabel)
    #
    # # // *-----------------------------------------------------------------* //
    # # // *--- Plot Peak-Picking results vs. Spectrogram ---*
    #
    # if plots > 0:
    #
    #     # // *-----------------------------------------------------------------* //
    #     # // *--- Perform the STFT ---*
    #
    #     NFFT = 2048
    #     ySTFT = stft(y, NFFT)
    #     assert (ySTFT.shape[1] == len(onset_env)), "Number of STFT frames != len onset_env"
    #
    #     # times_ch1 = frames_to_time(np.arange(len(onset_env_ch1)), fs, hop_length=512)
    #     # currently uses fixed hop_length
    #     times = frames_to_time(np.arange(len(onset_env)), sr, NFFT/4)
    #     plt.figure(facecolor='silver', edgecolor='k', figsize=(12, 8))
    #     ax = plt.subplot(2, 1, 1)
    #     specshow(amplitude_to_db(magphase(ySTFT)[0], ref=np.max), y_axis='log', x_axis='time', cmap=plt.cm.viridis)
    #     plt.title('CH1: Spectrogram (STFT)')
    #
    #     plt.subplot(2, 1, 2, sharex=ax)
    #     plt.plot(times, onset_env, alpha=0.66, label='Onset strength')
    #     plt.vlines(times[onsets], 0, onset_env.max(), color='r', alpha=0.8,
    #                                                    label='Selected peaks')
    #     plt.legend(frameon=True, framealpha=0.66)
    #     plt.axis('tight')
    #     plt.tight_layout()
    #
    #     plt.xlabel('time')
    #     plt.ylabel('Amplitude')
    #     plt.title('Onset Strength detection & Peak Selection')
    #
    # plt.show()

    return onsets_samples, onsets_time


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

def onset_detect(y=None, sr=48000, onset_envelope=None, hop_length=512,
                 backtrack=False, energy=None,
                 units='frames', **kwargs):
    
    """Basic onset detector.  Locate note onset events by picking peaks in an
    onset strength envelope.

    The `peak_pick` parameters were chosen by large-scale hyper-parameter
    optimization over the dataset provided by [1]_.

    .. [1] https://github.com/CPJKU/onset_db


    Parameters
    ----------
    y          : np.ndarray [shape=(n,)]
        audio time series

    sr         : number > 0 [scalar]
        sampling rate of `y`

    onset_envelope     : np.ndarray [shape=(m,)]
        (optional) pre-computed onset strength envelope

    hop_length : int > 0 [scalar]
        hop length (in samples)

    units : {'frames', 'samples', 'time'}
        The units to encode detected onset events in.
        By default, 'frames' are used.

    backtrack : bool
        If `True`, detected onset events are backtracked to the nearest
        preceding minimum of `energy`.

        This is primarily useful when using onsets as slice points for segmentation.

    energy : np.ndarray [shape=(m,)] (optional)
        An energy function to use for backtracking detected onset events.
        If none is provided, then `onset_envelope` is used.

    kwargs : additional keyword arguments
        Additional parameters for peak picking.

        See `librosa.util.peak_pick` for details.


    Returns
    -------

    onsets : np.ndarray [shape=(n_onsets,)]
        estimated positions of detected onsets, in whichever units
        are specified.  By default, frame indices.

        .. note::
            If no onset strength could be detected, onset_detect returns
            an empty list.


    Raises
    ------
    ParameterError
        if neither `y` nor `onsets` are provided

        or if `units` is not one of 'frames', 'samples', or 'time'

    See Also
    --------
    onset_strength : compute onset strength per-frame
    onset_backtrack : backtracking onset events
    librosa.util.peak_pick : pick peaks from a time series


    Examples
    --------
    Get onset times from a signal

    # >>> y, sr = librosa.load(librosa.util.example_audio_file(),
    # ...                      offset=30, duration=2.0)
    # >>> onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
    # >>> librosa.frames_to_time(onset_frames, sr=sr)
    array([ 0.07 ,  0.395,  0.511,  0.627,  0.766,  0.975,
            1.207,  1.324,  1.44 ,  1.788,  1.881])

    Or use a pre-computed onset envelope

    # >>> o_env = librosa.onset.onset_strength(y, sr=sr)
    # >>> times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    # >>> onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    #
    #
    # >>> import matplotlib.pyplot as plt
    # >>> D = np.abs(librosa.stft(y))
    # >>> plt.figure()
    # >>> ax1 = plt.subplot(2, 1, 1)
    # >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    # ...                          x_axis='time', y_axis='log')
    # >>> plt.title('Power spectrogram')
    # >>> plt.subplot(2, 1, 2, sharex=ax1)
    # >>> plt.plot(times, o_env, label='Onset strength')
    # >>> plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
    # ...            linestyle='--', label='Onsets')
    # >>> plt.axis('tight')
    # >>> plt.legend(frameon=True, framealpha=0.75)
    # >>> plt.show()

    """

    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError('y or onset_envelope must be provided')

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    onset_envelope -= onset_envelope.min()

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return np.array([], dtype=np.int)

    # Normalize onset strength function to [0, 1] range
    onset_envelope /= onset_envelope.max()

    # These parameter settings found by large-scale search
    kwargs.setdefault('pre_max', 0.03 * sr // hop_length)       # 30ms
    kwargs.setdefault('post_max', 0.00 * sr // hop_length + 1)  # 0ms
    kwargs.setdefault('pre_avg', 0.10 * sr // hop_length)       # 100ms
    kwargs.setdefault('post_avg', 0.10 * sr // hop_length + 1)  # 100ms
    kwargs.setdefault('wait', 0.03 * sr // hop_length)          # 30ms
    kwargs.setdefault('delta', 0.07)

    # Peak pick the onset envelope
    onsets = peak_pick(onset_envelope, **kwargs)

    # Optionally backtrack the events
    if backtrack:
        if energy is None:
            energy = onset_envelope

        onsets = onset_backtrack(onsets, energy)

    if units == 'frames':
        pass
    elif units == 'samples':
        onsets = frames_to_samples(onsets, hop_length=hop_length)
    elif units == 'time':
        onsets = frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError('Invalid unit type: {}'.format(units))

    return onsets


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

def onset_strength(
    y=None,
    sr=22050,
    S=None,
    lag=1,
    max_size=1,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    **kwargs
) -> np.ndarray:
    """Compute a spectral flux onset strength envelope.
    Onset strength at time ``t`` is determined by::
        mean_f max(0, S[f, t] - ref[f, t - lag])
    where ``ref`` is ``S`` after local max filtering along the frequency
    axis [#]_.
    By default, if a time series ``y`` is provided, S will be the
    log-power Mel spectrogram.
    .. [#] Böck, Sebastian, and Gerhard Widmer.
           "Maximum filter vibrato suppression for onset detection."
           16th International Conference on Digital Audio Effects,
           Maynooth, Ireland. 2013.
    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time-series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram
    lag : int > 0
        time lag for computing differences
    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.
    detrend : bool [scalar]
        Filter the onset strength to remove the DC component
    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.
    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``
    aggregate : function
        Aggregation function to use when combining onsets
        at different frequency bins.
        Default: `np.mean`
    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.
    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., m,)]
        vector containing the onset strength envelope.
        If the input contains multiple channels, then onset envelope is computed for each channel.
    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided
        or if ``lag`` or ``max_size`` are not positive integers
    See Also
    --------
    onset_detect
    onset_strength_multi
    Examples
    --------
    First, load some audio and plot the spectrogram
    # >>> import matplotlib.pyplot as plt
    # >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    # >>> D = np.abs(librosa.stft(y))
    # >>> times = librosa.times_like(D)
    # >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    # >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    # ...                          y_axis='log', x_axis='time', ax=ax[0])
    # >>> ax[0].set(title='Power spectrogram')
    # >>> ax[0].label_outer()
    # Construct a standard onset function
    # >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # >>> ax[1].plot(times, 2 + onset_env / onset_env.max(), alpha=0.8,
    # ...            label='Mean (mel)')
    # Median aggregation, and custom mel options
    # >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    # ...                                          aggregate=np.median,
    # ...                                          fmax=8000, n_mels=256)
    # >>> ax[1].plot(times, 1 + onset_env / onset_env.max(), alpha=0.8,
    # ...            label='Median (custom mel)')
    # Constant-Q spectrogram instead of Mel
    # >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    # >>> onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
    # >>> ax[1].plot(times, onset_env / onset_env.max(), alpha=0.8,
    # ...          label='Mean (CQT)')
    # >>> ax[1].legend()
    # >>> ax[1].set(ylabel='Normalized strength', yticks=[])
    """

    if aggregate is False:
        raise ParameterError(
            f"aggregate parameter cannot be False when computing full-spectrum onset strength."
        )

    odf_all = onset_strength_multi(
        y=y,
        sr=sr,
        S=S,
        lag=lag,
        max_size=max_size,
        detrend=detrend,
        center=center,
        feature=feature,
        aggregate=aggregate,
        channels=None,
        **kwargs,
    )

    return odf_all[..., 0, :]


def onset_backtrack(events: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Backtrack detected onset events to the nearest preceding local
    minimum of an energy function.
    This function can be used to roll back the timing of detected onsets
    from a detected peak amplitude to the preceding minimum.
    This is most useful when using onsets to determine slice points for
    segmentation, as described by [#]_.
    .. [#] Jehan, Tristan.
           "Creating music by listening"
           Doctoral dissertation
           Massachusetts Institute of Technology, 2005.
    Parameters
    ----------
    events : np.ndarray, dtype=int
        List of onset event frame indices, as computed by `onset_detect`
    energy : np.ndarray, shape=(m,)
        An energy function
    Returns
    -------
    events_backtracked : np.ndarray, shape=events.shape
        The input events matched to nearest preceding minima of ``energy``.
    Examples
    --------
    Backtrack the events using the onset envelope
    >>> y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    >>> oenv = librosa.onset.onset_strength(y=y, sr=sr)
    >>> times = librosa.times_like(oenv)
    >>> # Detect events without backtracking
    >>> onset_raw = librosa.onset.onset_detect(onset_envelope=oenv,
    ...                                        backtrack=False)
    >>> onset_bt = librosa.onset.onset_backtrack(onset_raw, oenv)
    Backtrack the events using the RMS values
    >>> S = np.abs(librosa.stft(y=y))
    >>> rms = librosa.feature.rms(S=S)
    >>> onset_bt_rms = librosa.onset.onset_backtrack(onset_raw, rms[0])
    Plot the results
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(nrows=3, sharex=True)
    >>> librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[0])
    >>> ax[0].label_outer()
    >>> ax[1].plot(times, oenv, label='Onset strength')
    >>> ax[1].vlines(librosa.frames_to_time(onset_raw), 0, oenv.max(), label='Raw onsets')
    >>> ax[1].vlines(librosa.frames_to_time(onset_bt), 0, oenv.max(), label='Backtracked', color='r')
    >>> ax[1].legend()
    >>> ax[1].label_outer()
    >>> ax[2].plot(times, rms[0], label='RMS')
    >>> ax[2].vlines(librosa.frames_to_time(onset_bt_rms), 0, rms.max(), label='Backtracked (RMS)', color='r')
    >>> ax[2].legend()
    """

    # Find points where energy is non-increasing
    # all points:  energy[i] <= energy[i-1]
    # tail points: energy[i] < energy[i+1]
    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:]))

    # Pad on a 0, just in case we have onsets with no preceding minimum
    # Shift by one to account for slicing in minima detection
    minima = fix_frames(1 + minima, x_min=0)

    # Only match going left from the detected events
    results: np.ndarray = minima[match_events(events, minima, right=False)]
    return results


@cache(level=30)
def onset_strength_multi(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    lag=1,
    max_size=1,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    channels=None,
    **kwargs
) -> np.ndarray:
    """Compute a spectral flux onset strength envelope across multiple channels.
    Onset strength for channel ``i`` at time ``t`` is determined by::
        mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])
    Parameters
    ----------
    y : np.ndarray [shape=(..., n,)]
        audio time-series. Multi-channel is supported.
    sr : number > 0 [scalar]
        sampling rate of ``y``
    S : np.ndarray [shape=(..., d, m)]
        pre-computed (log-power) spectrogram
    n_fft : int > 0 [scalar]
        FFT window size for use in ``feature()`` if ``S`` is not provided.
    hop_length : int > 0 [scalar]
        hop length for use in ``feature()`` if ``S`` is not provided.
    lag : int > 0
        time lag for computing differences
    max_size : int > 0
        size (in frequency bins) of the local max filter.
        set to `1` to disable filtering.
    ref : None or np.ndarray [shape=(d, m)]
        An optional pre-computed reference spectrum, of the same shape as ``S``.
        If not provided, it will be computed from ``S``.
        If provided, it will override any local max filtering governed by ``max_size``.
    detrend : bool [scalar]
        Filter the onset strength to remove the DC component
    center : bool [scalar]
        Shift the onset function by ``n_fft // (2 * hop_length)`` frames.
        This corresponds to using a centered frame analysis in the short-time Fourier
        transform.
    feature : function
        Function for computing time-series features, eg, scaled spectrograms.
        By default, uses `librosa.feature.melspectrogram` with ``fmax=sr/2``
        Must support arguments: ``y, sr, n_fft, hop_length``
    aggregate : function or False
        Aggregation function to use when combining onsets
        at different frequency bins.
        If ``False``, then no aggregation is performed.
        Default: `np.mean`
    channels : list or None
        Array of channel boundaries or slice objects.
        If `None`, then a single channel is generated to span all bands.
    **kwargs : additional keyword arguments
        Additional parameters to ``feature()``, if ``S`` is not provided.
    Returns
    -------
    onset_envelope : np.ndarray [shape=(..., n_channels, m)]
        array containing the onset strength envelope for each specified channel
    Raises
    ------
    ParameterError
        if neither ``(y, sr)`` nor ``S`` are provided
    See Also
    --------
    onset_strength
    Notes
    -----
    This function caches at level 30.
    Examples
    --------
    First, load some audio and plot the spectrogram
    # >>> import matplotlib.pyplot as plt
    # >>> y, sr = librosa.load(librosa.ex('choice'), duration=5)
    # >>> D = np.abs(librosa.stft(y))
    # >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    # >>> img1 = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    # ...                          y_axis='log', x_axis='time', ax=ax[0])
    # >>> ax[0].set(title='Power spectrogram')
    # >>> ax[0].label_outer()
    # >>> fig.colorbar(img1, ax=[ax[0]], format="%+2.f dB")
    # Construct a standard onset function over four sub-bands
    # >>> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
    # ...                                                     channels=[0, 32, 64, 96, 128])
    # >>> img2 = librosa.display.specshow(onset_subbands, x_axis='time', ax=ax[1])
    # >>> ax[1].set(ylabel='Sub-bands', title='Sub-band onset strength')
    # >>> fig.colorbar(img2, ax=[ax[1]])
    """

    if feature is None:
        feature = melspectrogram
        kwargs.setdefault("fmax", 0.5 * sr)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, (int, np.integer)):
        raise ParameterError(f"lag={lag} must be a positive integer")

    if max_size < 1 or not isinstance(max_size, (int, np.integer)):
        raise ParameterError(f"max_size={max_size} must be a positive integer")

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))

        # Convert to dBs
        S = power_to_db(S)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if max_size == 1:
        ref = S
    else:
        ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)

    # Compute difference to the reference, spaced by lag
    onset_env = S[..., lag:] - ref[..., :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    if callable(aggregate):
        onset_env = sync(
            onset_env, channels, aggregate=aggregate, pad=pad, axis=-2
        )

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    padding = [(0, 0) for _ in onset_env.shape]
    padding[-1] = (int(pad_width), 0)
    onset_env = np.pad(onset_env, padding, mode="constant")

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[..., : S.shape[-1]]

    return onset_env


# @cache(level=30)
# def onset_strength_multi(y=None, sr=48000, S=None, n_fft=2048, hop_length=512, lag=1,
#                          max_size=1, detrend=False, center=True, feature=None,
#                          aggregate=None, channels=None, **kwargs):
#     """Compute a spectral flux onset strength envelope across multiple channels.
#
#     Onset strength for channel `i` at time `t` is determined by:
#
#     `mean_{f in channels[i]} max(0, S[f, t+1] - S[f, t])`
#
#
#     Parameters
#     ----------
#     y        : np.ndarray [shape=(n,)]
#         audio time-series
#
#     sr       : number > 0 [scalar]
#         sampling rate of `y`
#
#     S        : np.ndarray [shape=(d, m)]
#         pre-computed (log-power) spectrogram
#
#     n_fft : int > 0 [scaler]
#         FFT window size for use in ``feature()`` if ``s`` is not provided
#
#     hop_length : int > 0 [scaler]
#         hop length for use in ``feature()`` if ``s`` is not provided
#
#     lag      : int > 0
#         time lag for computing differences
#
#     max_size : int > 0
#         size (in frequency bins) of the local max filter.
#         set to `1` to disable filtering.
#
#     detrend : bool [scalar]
#         Filter the onset strength to remove the DC component
#
#     center : bool [scalar]
#         Shift the onset function by `n_fft / (2 * hop_length)` frames
#
#     feature : function
#         Function for computing time-series features, eg, scaled spectrograms.
#         By default, uses `librosa.feature.melspectrogram` with `fmax=11025.0`
#
#     aggregate : function
#         Aggregation function to use when combining onsets
#         at different frequency bins.
#
#         Default: `np.mean`
#
#     channels : list or None
#         Array of channel boundaries or slice objects.
#         If `None`, then a single channel is generated to span all bands.
#
#     kwargs : additional keyword arguments
#         Additional parameters to `feature()`, if `S` is not provided.
#
#
#     Returns
#     -------
#     onset_envelope   : np.ndarray [shape=(n_channels, m)]
#         array containing the onset strength envelope for each specified channel
#
#
#     Raises
#     ------
#     ParameterError
#         if neither `(y, sr)` nor `S` are provided
#
#
#     See Also
#     --------
#     onset_strength
#
#     Notes
#     -----
#     This function caches at level 30.
#
#     Examples
#     --------
#     First, load some audio and plot the spectrogram
#
#     >> import matplotlib.pyplot as plt
#     >> y, sr = librosa.load(librosa.util.example_audio_file(),
#     ...                      duration=10.0)
#     >> D = np.abs(librosa.stft(y))
#     >> plt.figure()
#     >> plt.subplot(2, 1, 1)
#     >> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
#     ...                          y_axis='log')
#     >> plt.title('Power spectrogram')
#
#     Construct a standard onset function over four sub-bands
#
#     >> onset_subbands = librosa.onset.onset_strength_multi(y=y, sr=sr,
#     ...                                                     channels=[0, 32, 64, 96, 128])
#     >> plt.subplot(2, 1, 2)
#     >> librosa.display.specshow(onset_subbands, x_axis='time')
#     >> plt.ylabel('Sub-bands')
#     >> plt.title('Sub-band onset strength')
#
#     """
#
#     if feature is None:
#         feature = melspectrogram
#         kwargs.setdefault('fmax', 0.5 * sr)
#
#     if aggregate is None:
#         aggregate = np.mean
#
#     if lag < 1 or not isinstance(lag, (int, np.integer)):
#         raise ParameterError('lag must be a positive integer')
#
#     if max_size < 1 or not isinstance(max_size, (int, np.integer)):
#         raise ParameterError('max_size must be a positive integer')
#
#     # First, compute mel spectrogram
#     if S is None:
#         S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))
#
#         # Convert to dBs
#         S = power_to_db(S)
#
#     # Retrieve the n_fft and hop_length,
#     # or default values for onsets if not provided
#     n_fft = kwargs.get('n_fft', 2048)
#     hop_length = kwargs.get('hop_length', 512)
#
#     # Ensure that S is at least 2-d
#     S = np.atleast_2d(S)
#
#     # Compute the reference spectrogram.
#     # Efficiency hack: skip filtering step and pass by reference
#     # if max_size will produce a no-op.
#     if max_size == 1:
#         ref_spec = S
#     else:
#         ref_spec = scipy.ndimage.maximum_filter1d(S, max_size, axis=-2)
#
#     # Compute difference to the reference, spaced by lag
#     onset_env = S[:, lag:] - ref_spec[:, :-lag]
#
#     # Discard negatives (decreasing amplitude)
#     onset_env = np.maximum(0.0, onset_env)
#
#     # Aggregate within channels
#     pad = True
#     if channels is None:
#         channels = [slice(None)]
#     else:
#         pad = False
#
#     if callable(aggregate):
#         onset_env = sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=-2)
#
#     # compensate for lag
#     pad_width = lag
#     if center:
#         # Counter-act framing effects. Shift the onsets by n_fft / hop_length
#         pad_width += n_fft // (2 * hop_length)
#
#     padding = [(0, 0) for _ in onset_env.shape]
#     padding[-1] = (int(pad_width), 0)
#     onset_env = np.pad(onset_env, padding, mode="constant")
#
#     # remove the DC component
#     if detrend:
#         onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99],
#                                          onset_env, axis=-1)
#
#     # Trim to match the input duration
#     if center:
#         onset_env = onset_env[:, :S.shape[-1]]
#
#     return onset_env


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //

def get_peak_regions(peaks, length):
    ''' returns an array of peak regions (number of samples between peaks '''
        
    peak_regions = np.zeros((len(peaks)+1))
    for i in range(len(peaks)+1):
        if i == 0:
            peak_regions[0] = peaks[0]
        elif i == len(peaks):
            peak_regions[i] = length - peaks[i-1]
        else:
            peak_regions[i] = peaks[i] - peaks[i-1]
            
    return peak_regions


def getOnsetSampleSegments(onsets_samples, totalSamples):
    ''' returns an array of peak regions (number of samples between peaks '''
        
    onset_sample_segments = np.zeros((len(onsets_samples)+1))
    for i in range(len(onsets_samples)+1):
        if i == 0:
            onset_sample_segments[0] = onsets_samples[0]
        elif i == len(onsets_samples):
            onset_sample_segments[i] = totalSamples - onsets_samples[i-1]
        else:
            onset_sample_segments[i] = onsets_samples[i] - onsets_samples[i-1]
            
    return onset_sample_segments


def getOnsetTimeSegments(onsets_time, totalTime):
    ''' returns an array of peak regions (number of samples between peaks '''
        
    onset_time_segments = np.zeros((len(onsets_time)+1))
    for i in range(len(onsets_time)+1):
        if i == 0:
            onset_time_segments[0] = onsets_time[0]
        elif i == len(onsets_time):
            onset_time_segments[i] = totalTime - onsets_time[i-1]
        else:
            onset_time_segments[i] = onsets_time[i] - onsets_time[i-1]
            
    return onset_time_segments