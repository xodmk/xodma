# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodmaSpectralUtil.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
# XODMK Audio Tools - Time and frequency utilities
#
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import re
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy as sp
# import scipy.signal
import six


currentDir = os.getcwd()
rootDir = os.path.dirname(currentDir)
sys.path.insert(0, rootDir+'/xodma')

from cache import cache
from xodmaParameterError import ParameterError
# from xodmaAudioTools import valid_audio


# temp python debugger - use >>>pdb.set_trace() to set break
# import pdb


# // *---------------------------------------------------------------------* //

# Constrain STFT block sizes to 256 KB
# ** note: pymalloc allocates memory in 256 kB chunks, called arenas
MAX_MEM_BLOCK = 2**8 * 2**10


__all__ = ['MAX_MEM_BLOCK', 'frame',
           'frames_to_samples', 'frames_to_time',
           'samples_to_frames', 'samples_to_time',
           'time_to_samples', 'time_to_frames',
           'note_to_hz', 'note_to_midi',
           'midi_to_hz', 'midi_to_note',
           'hz_to_note', 'hz_to_midi',
           'hz_to_mel', 'hz_to_octs',
           'mel_to_hz', 'octs_to_hz',
           'fft_frequencies',
           'cqt_frequencies',
           'mel_frequencies',
           'tempo_frequencies',
           'A_weighting',
           'window_bandwidth',
           'get_window']
           

# Dictionary of window function bandwidths

WINDOW_BANDWIDTHS = {'bart': 1.3334961334912805,
                     'barthann': 1.4560255965133932,
                     'bartlett': 1.3334961334912805,
                     'bkh': 2.0045975283585014,
                     'black': 1.7269681554262326,
                     'blackharr': 2.0045975283585014,
                     'blackman': 1.7269681554262326,
                     'blackmanharris': 2.0045975283585014,
                     'blk': 1.7269681554262326,
                     'bman': 1.7859588613860062,
                     'bmn': 1.7859588613860062,
                     'bohman': 1.7859588613860062,
                     'box': 1.0,
                     'boxcar': 1.0,
                     'brt': 1.3334961334912805,
                     'brthan': 1.4560255965133932,
                     'bth': 1.4560255965133932,
                     'cosine': 1.2337005350199792,
                     'flat': 2.7762255046484143,
                     'flattop': 2.7762255046484143,
                     'flt': 2.7762255046484143,
                     'halfcosine': 1.2337005350199792,
                     'ham': 1.3629455320350348,
                     'hamm': 1.3629455320350348,
                     'hamming': 1.3629455320350348,
                     'han': 1.50018310546875,
                     'hann': 1.50018310546875,
                     'hanning': 1.50018310546875,
                     'nut': 1.9763500280946082,
                     'nutl': 1.9763500280946082,
                     'nuttall': 1.9763500280946082,
                     'ones': 1.0,
                     'par': 1.9174603174603191,
                     'parz': 1.9174603174603191,
                     'parzen': 1.9174603174603191,
                     'rect': 1.0,
                     'rectangular': 1.0,
                     'tri': 1.3331706523555851,
                     'triang': 1.3331706523555851,
                     'triangle': 1.3331706523555851}


def frame(
    x: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    axis: int = -1,
    writeable: bool = False,
    subok: bool = False,
) -> np.ndarray:
    """Slice a data array into (overlapping) frames.
    This implementation uses low-level stride manipulation to avoid
    making a copy of the data.  The resulting frame representation
    is a new view of the same input data.
    For example, a one-dimensional input ``x = [0, 1, 2, 3, 4, 5, 6]``
    can be framed with frame length 3 and hop length 2 in two ways.
    The first (``axis=-1``), results in the array ``x_frames``::
        [[0, 2, 4],
         [1, 3, 5],
         [2, 4, 6]]
    where each column ``x_frames[:, i]`` contains a contiguous slice of
    the input ``x[i * hop_length : i * hop_length + frame_length]``.
    The second way (``axis=0``) results in the array ``x_frames``::
        [[0, 1, 2],
         [2, 3, 4],
         [4, 5, 6]]
    where each row ``x_frames[i]`` contains a contiguous slice of the input.
    This generalizes to higher dimensional inputs, as shown in the examples below.
    In general, the framing operation increments by 1 the number of dimensions,
    adding a new "frame axis" either before the framing axis (if ``axis < 0``)
    or after the framing axis (if ``axis >= 0``).
    Parameters
    ----------
    x : np.ndarray
        Array to frame
    frame_length : int > 0 [scalar]
        Length of the frame
    hop_length : int > 0 [scalar]
        Number of steps to advance between frames
    axis : int
        The axis along which to frame.
    writeable : bool
        If ``True``, then the framed view of ``x`` is read-only.
        If ``False``, then the framed view is read-write.  Note that writing to the framed view
        will also write to the input array ``x`` in this case.
    subok : bool
        If True, sub-classes will be passed-through, otherwise the returned array will be
        forced to be a base-class array (default).
    Returns
    -------
    x_frames : np.ndarray [shape=(..., frame_length, N_FRAMES, ...)]
        A framed view of ``x``, for example with ``axis=-1`` (framing on the last dimension)::
            x_frames[..., j] == x[..., j * hop_length : j * hop_length + frame_length]
        If ``axis=0`` (framing on the first dimension), then::
            x_frames[j] = x[j * hop_length : j * hop_length + frame_length]
    Raises
    ------
    ParameterError
        If ``x.shape[axis] < frame_length``, there is not enough data to fill one frame.
        If ``hop_length < 1``, frames cannot advance.
    See Also
    --------
    numpy.lib.stride_tricks.as_strided
    Examples
    --------
    # Extract 2048-sample frames from monophonic signal with a hop of 64 samples per frame
    # >>> y, sr = librosa.load(librosa.ex('trumpet'))
    # >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    # >>> frames
    # array([[-1.407e-03, -2.604e-02, ..., -1.795e-05, -8.108e-06],
    #        [-4.461e-04, -3.721e-02, ..., -1.573e-05, -1.652e-05],
    #        ...,
    #        [ 7.960e-02, -2.335e-01, ..., -6.815e-06,  1.266e-05],
    #        [ 9.568e-02, -1.252e-01, ...,  7.397e-06, -1.921e-05]],
    #       dtype=float32)
    # >>> y.shape
    # (117601,)
    # >>> frames.shape
    # (2048, 1806)
    # Or frame along the first axis instead of the last:
    # >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64, axis=0)
    # >>> frames.shape
    # (1806, 2048)
    # Frame a stereo signal:
    # >>> y, sr = librosa.load(librosa.ex('trumpet', hq=True), mono=False)
    # >>> y.shape
    # (2, 117601)
    # >>> frames = librosa.util.frame(y, frame_length=2048, hop_length=64)
    # (2, 2048, 1806)
    # Carve an STFT into fixed-length patches of 32 frames with 50% overlap
    # >>> y, sr = librosa.load(librosa.ex('trumpet'))
    # >>> S = np.abs(librosa.stft(y))
    # >>> S.shape
    # (1025, 230)
    # >>> S_patch = librosa.util.frame(S, frame_length=32, hop_length=16)
    # >>> S_patch.shape
    # (1025, 32, 13)
    # >>> # The first patch contains the first 32 frames of S
    # >>> np.allclose(S_patch[:, :, 0], S[:, :32])
    # True
    # >>> # The second patch contains frames 16 to 16+32=48, and so on
    # >>> np.allclose(S_patch[:, :, 1], S[:, 16:48])
    # True
    """

    # This implementation is derived from numpy.lib.stride_tricks.sliding_window_view (1.20.0)
    # https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.sliding_window_view.html

    x = np.array(x, copy=False, subok=subok)

    if x.shape[axis] < frame_length:
        raise ParameterError(
            f"Input is too short (n={x.shape[axis]:d}) for frame_length={frame_length:d}"
        )

    if hop_length < 1:
        raise ParameterError(f"Invalid hop_length: {hop_length:d}")

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )

    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1

    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    return xw[tuple(slices)]


def frames_to_samples(frames, hop_length=512, n_fft=None):
    """Converts frame indices to audio sample indices

    Parameters
    ----------
    frames     : np.ndarray [shape=(n,)]
        vector of frame indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `n_fft / 2`
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in samples) of each given frame number:
        `times[i] = frames[i] * hop_length`

    See Also
    --------
    frames_to_time : convert frame indices to time values
    samples_to_frames : convert sample indices to frame indices

    Examples
    --------
    >> y, sr = librosa.load(librosa.util.example_audio_file())
    >> tempo, beats = librosa.beat.beat_track(y, sr=sr)
    >> beat_samples = librosa.frames_to_samples(beats)
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    return (np.atleast_1d(frames) * hop_length + offset).astype(int)


def samples_to_frames(samples, hop_length=512, n_fft=None):
    """Converts sample indices into STFT frames.

    Examples
    --------
    >> # Get the frame numbers for every 256 samples
    >> samples_to_frames(np.arange(0, 48000, 256))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,
            7,  7,  8,  8,  9,  9, 10, 10, 11, 11, 12, 12, 13, 13,
           14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20,
           21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27,
           28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,
           35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41,
           42, 42, 43])

    Parameters
    ----------
    samples : np.ndarray [shape=(n,)]
        vector of sample indices

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `- n_fft / 2`
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times:
        `frames[i] = floor( samples[i] / hop_length )`

    See Also
    --------
    samples_to_time : convert sample indices to time values
    frames_to_samples : convert frame indices to sample indices
    """

    offset = 0
    if n_fft is not None:
        offset = int(n_fft // 2)

    samples = np.atleast_1d(samples)
    return np.floor((samples - offset) // hop_length).astype(int)


def frames_to_time(frames, sr=48000, hop_length=512, n_fft=None):
    """Converts frame counts to time (seconds)

    Parameters
    ----------
    frames     : np.ndarray [shape=(n,)]
        vector of frame numbers

    sr         : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `n_fft / 2`
        to counteract windowing effects when using a non-centered STFT.

    Returns
    -------
    times : np.ndarray [shape=(n,)]
        time (in seconds) of each given frame number:
        `times[i] = frames[i] * hop_length / sr`

    See Also
    --------
    time_to_frames : convert time values to frame indices
    frames_to_samples : convert frame indices to sample indices

    Examples
    --------
    >> y, sr = librosa.load(librosa.util.example_audio_file())
    >> tempo, beats = librosa.beat.beat_track(y, sr=sr)
    >> beat_times = librosa.frames_to_time(beats, sr=sr)
    """

    samples = frames_to_samples(frames,
                                hop_length=hop_length,
                                n_fft=n_fft)

    return samples_to_time(samples, sr=sr)


def time_to_frames(times, sr=48000, hop_length=512, n_fft=None):
    """Converts time stamps into STFT frames.

    Parameters
    ----------
    times : np.ndarray [shape=(n,)]
        vector of time stamps

    sr : number > 0 [scalar]
        audio sampling rate

    hop_length : int > 0 [scalar]
        number of samples between successive frames

    n_fft : None or int > 0 [scalar]
        Optional: length of the FFT window.
        If given, time conversion will include an offset of `- n_fft / 2`
        to counteract windowing effects in STFT.

        .. note:: This may result in negative frame indices.

    Returns
    -------
    frames : np.ndarray [shape=(n,), dtype=int]
        Frame numbers corresponding to the given times:
        `frames[i] = floor( times[i] * sr / hop_length )`

    See Also
    --------
    frames_to_time : convert frame indices to time values
    time_to_samples : convert time values to sample indices

    Examples
    --------
    Get the frame numbers for every 100ms

    >> librosa.time_to_frames(np.arange(0, 1, 0.1),
    ...                         sr=48000, hop_length=512)
    array([ 0,  4,  8, 12, 17, 21, 25, 30, 34, 38])

    """

    samples = time_to_samples(times, sr=sr)

    return samples_to_frames(samples, hop_length=hop_length, n_fft=n_fft)


def time_to_samples(times, sr=48000):
    '''Convert timestamps (in seconds) to sample indices.

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
    >> librosa.time_to_samples(np.arange(0, 1, 0.1), sr=48000)
    array([    0,  2205,  4410,  6615,  8820, 11025, 13230, 15435,
           17640, 19845])

    '''

    return (np.atleast_1d(times) * sr).astype(int)


def samples_to_time(samples, sr=48000):
    '''Convert sample indices to time (in seconds).

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

    >> librosa.samples_to_time(np.arange(0, 48000, 512))
    array([ 0.   ,  0.023,  0.046,  0.07 ,  0.093,  0.116,  0.139,
            0.163,  0.186,  0.209,  0.232,  0.255,  0.279,  0.302,
            0.325,  0.348,  0.372,  0.395,  0.418,  0.441,  0.464,
            0.488,  0.511,  0.534,  0.557,  0.58 ,  0.604,  0.627,
            0.65 ,  0.673,  0.697,  0.72 ,  0.743,  0.766,  0.789,
            0.813,  0.836,  0.859,  0.882,  0.906,  0.929,  0.952,
            0.975,  0.998])
    '''

    return np.atleast_1d(samples) / float(sr)


def note_to_hz(note, **kwargs):
    '''Convert one or more note names to frequency (Hz)

    Examples
    --------
    >> # Get the frequency of a note
    >> librosa.note_to_hz('C')
    array([ 16.352])
    >> # Or multiple notes
    >> librosa.note_to_hz(['A3', 'A4', 'A5'])
    array([ 220.,  440.,  880.])
    >> # Or notes with tuning deviations
    >> librosa.note_to_hz('C2-32', round_midi=False)
    array([ 64.209])

    Parameters
    ----------
    note : str or iterable of str
        One or more note names to convert

    kwargs : additional keyword arguments
        Additional parameters to `note_to_midi`

    Returns
    -------
    frequencies : np.ndarray [shape=(len(note),)]
        Array of frequencies (in Hz) corresponding to `note`

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    '''
    return midi_to_hz(note_to_midi(note, **kwargs))


def note_to_midi(note, round_midi=True):
    '''Convert one or more spelled notes to MIDI number(s).

    Notes may be spelled out with optional accidentals or octave numbers.

    The leading note name is case-insensitive.

    Sharps are indicated with ``#``, flats may be indicated with ``!`` or ``b``.

    Parameters
    ----------
    note : str or iterable of str
        One or more note names.

    round_midi : bool
        - If `True`, allow for fractional midi notes
        - Otherwise, round cent deviations to the nearest note

    Returns
    -------
    midi : float or np.array
        Midi note numbers corresponding to inputs.

    Raises
    ------
    ParameterError
        If the input is not in valid note format

    See Also
    --------
    midi_to_note
    note_to_hz

    Examples
    --------
    >> librosa.note_to_midi('C')
    12
    >> librosa.note_to_midi('C#3')
    49
    >> librosa.note_to_midi('f4')
    65
    >> librosa.note_to_midi('Bb-1')
    10
    >> librosa.note_to_midi('A!8')
    116
    >> # Lists of notes also work
    >> librosa.note_to_midi(['C', 'E', 'G'])
    array([12, 16, 19])

    '''

    if not isinstance(note, six.string_types):
        return np.array([note_to_midi(n, round_midi=round_midi) for n in note])

    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1}

    match = re.match(r'^(?P<note>[A-Ga-g])'
                     r'(?P<accidental>[#b!]*)'
                     r'(?P<octave>[+-]?\d+)?'
                     r'(?P<cents>[+-]\d+)?$',
                     note)
    if not match:
        raise ParameterError('Improper note format: {:s}'.format(note))

    pitch = match.group('note').upper()
    offset = np.sum([acc_map[o] for o in match.group('accidental')])
    octave = match.group('octave')
    cents = match.group('cents')

    if not octave:
        octave = 0
    else:
        octave = int(octave)

    if not cents:
        cents = 0
    else:
        cents = int(cents) * 1e-2

    note_value = 12 * (octave + 1) + pitch_map[pitch] + offset + cents

    if round_midi:
        note_value = int(np.round(note_value))

    return note_value


def midi_to_note(midi, octave=True, cents=False):
    '''Convert one or more MIDI numbers to note strings.

    MIDI numbers will be rounded to the nearest integer.

    Notes will be of the format 'C0', 'C#0', 'D0', ...

    Examples
    --------
    >> librosa.midi_to_note(0)
    'C-1'
    >> librosa.midi_to_note(37)
    'C#2'
    >> librosa.midi_to_note(-2)
    'A#-2'
    >> librosa.midi_to_note(104.7)
    'A7'
    >> librosa.midi_to_note(104.7, cents=True)
    'A7-30'
    >> librosa.midi_to_note(list(range(12, 24)))
    ['C0', 'C#0', 'D0', 'D#0', 'E0', 'F0', 'F#0', 'G0', 'G#0', 'A0', 'A#0', 'B0']

    Parameters
    ----------
    midi : int or iterable of int
        Midi numbers to convert.

    octave: bool
        If True, include the octave number

    cents: bool
        If true, cent markers will be appended for fractional notes.
        Eg, `midi_to_note(69.3, cents=True)` == `A4+03`

    Returns
    -------
    notes : str or iterable of str
        Strings describing each midi note.

    Raises
    ------
    ParameterError
        if `cents` is True and `octave` is False

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    '''

    if cents and not octave:
        raise ParameterError('Cannot encode cents without octave information.')

    if not np.isscalar(midi):
        return [midi_to_note(x, octave=octave, cents=cents) for x in midi]

    note_map = ['C', 'C#', 'D', 'D#',
                'E', 'F', 'F#', 'G',
                'G#', 'A', 'A#', 'B']

    note_num = int(np.round(midi))
    note_cents = int(100 * np.around(midi - note_num, 2))

    note = note_map[note_num % 12]

    if octave:
        note = '{:s}{:0d}'.format(note, int(note_num / 12) - 1)
    if cents:
        note = '{:s}{:+02d}'.format(note, note_cents)

    return note


def midi_to_hz(notes):
    """Get the frequency (Hz) of MIDI note(s)

    Examples
    --------
    >> librosa.midi_to_hz(36)
    array([ 65.406])

    >> librosa.midi_to_hz(np.arange(36, 48))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,
             87.307,   92.499,   97.999,  103.826,  110.   ,
            116.541,  123.471])

    Parameters
    ----------
    notes       : int or np.ndarray [shape=(n,), dtype=int]
        midi number(s) of the note(s)

    Returns
    -------
    frequency   : np.ndarray [shape=(n,), dtype=float]
        frequency (frequencies) of `notes` in Hz

    See Also
    --------
    hz_to_midi
    note_to_hz
    """

    return 440.0 * (2.0 ** ((np.atleast_1d(notes) - 69.0)/12.0))


def hz_to_midi(frequencies):
    """Get the closest MIDI note number(s) for given frequencies

    Examples
    --------
    >> librosa.hz_to_midi(60)
    array([ 34.506])
    >> librosa.hz_to_midi([110, 220, 440])
    array([ 45.,  57.,  69.])

    Parameters
    ----------
    frequencies   : float or np.ndarray [shape=(n,), dtype=float]
        frequencies to convert

    Returns
    -------
    note_nums     : np.ndarray [shape=(n,), dtype=int]
        closest MIDI notes to `frequencies`

    See Also
    --------
    midi_to_hz
    note_to_midi
    hz_to_note
    """

    return 12 * (np.log2(np.atleast_1d(frequencies)) - np.log2(440.0)) + 69


def hz_to_note(frequencies, **kwargs):
    '''Convert one or more frequencies (in Hz) to the nearest note names.

    Parameters
    ----------
    frequencies : float or iterable of float
        Input frequencies, specified in Hz

    kwargs : additional keyword arguments
        Arguments passed through to `midi_to_note`


    Returns
    -------
    notes : list of str
        `notes[i]` is the closest note name to `frequency[i]`
        (or `frequency` if the input is scalar)


    See Also
    --------
    hz_to_midi
    midi_to_note
    note_to_hz


    Examples
    --------
    Get a single note name for a frequency

    >> librosa.hz_to_note(440.0)
    ['A5']

    Get multiple notes with cent deviation

    >> librosa.hz_to_note([32, 64], cents=True)
    ['C1-38', 'C2-38']

    Get multiple notes, but suppress octave labels

    >> librosa.hz_to_note(440.0 * (2.0 ** np.linspace(0, 1, 12)),
    ...                    octave=False)
    ['A', 'A#', 'B', 'C', 'C#', 'D', 'E', 'F', 'F#', 'G', 'G#', 'A']

    '''
    return midi_to_note(hz_to_midi(frequencies), **kwargs)


def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels

    Examples
    --------
    >> librosa.hz_to_mel(60)
    array([ 0.9])
    >> librosa.hz_to_mel([110, 220, 440])
    array([ 1.65,  3.3 ,  6.6 ])

    Parameters
    ----------
    frequencies   : np.ndarray [shape=(n,)] , float
        scalar or array of frequencies
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    mels        : np.ndarray [shape=(n,)]
        input frequencies in Mels

    See Also
    --------
    mel_to_hz
    """

    frequencies = np.asanyarray(frequencies)

    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if frequencies.ndim:
        # If we have array data, vectorize
        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel + np.log(frequencies[log_t]/min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep

    return mels


def mel_to_hz(mels, htk=False):
    """Convert mel bin numbers to frequencies

    Examples
    --------
    >> librosa.mel_to_hz(3)
    array([ 200.])

    >> librosa.mel_to_hz([1,2,3,4,5])
    array([  66.667,  133.333,  200.   ,  266.667,  333.333])

    Parameters
    ----------
    mels          : np.ndarray [shape=(n,)], float
        mel bins to convert
    htk           : bool
        use HTK formula instead of Slaney

    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        input mels in Hz

    See Also
    --------
    hz_to_mel
    """

    mels = np.asanyarray(mels)

    if htk:
        return 700.0 * (10.0**(mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0                         # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
    logstep = np.log(6.4) / 27.0                # step size for log region

    if mels.ndim:
        # If we have vector data, vectorize
        log_t = (mels >= min_log_mel)
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))

    return freqs


def hz_to_octs(frequencies, A440=440.0):
    """Convert frequencies (Hz) to (fractional) octave numbers.

    Examples
    --------
    >> librosa.hz_to_octs(440.0)
    array([ 4.])
    >> librosa.hz_to_octs([32, 64, 128, 256])
    array([ 0.219,  1.219,  2.219,  3.219])

    Parameters
    ----------
    frequencies   : np.ndarray [shape=(n,)] or float
        scalar or vector of frequencies
    A440          : float
        frequency of A440 (in Hz)

    Returns
    -------
    octaves       : np.ndarray [shape=(n,)]
        octave number for each frequency

    See Also
    --------
    octs_to_hz
    """
    return np.log2(np.atleast_1d(frequencies) / (float(A440) / 16))


def octs_to_hz(octs, A440=440.0):
    """Convert octaves numbers to frequencies.

    Octaves are counted relative to A.

    Examples
    --------
    >> librosa.octs_to_hz(1)
    array([ 55.])
    >> librosa.octs_to_hz([-2, -1, 0, 1, 2])
    array([   6.875,   13.75 ,   27.5  ,   55.   ,  110.   ])

    Parameters
    ----------
    octaves       : np.ndarray [shape=(n,)] or float
        octave number for each frequency
    A440          : float
        frequency of A440

    Returns
    -------
    frequencies   : np.ndarray [shape=(n,)]
        scalar or vector of frequencies

    See Also
    --------
    hz_to_octs
    """
    return (float(A440) / 16)*(2.0**np.atleast_1d(octs))


def fft_frequencies(sr=48000, n_fft=2048):
    '''Alternative implementation of `np.fft.fftfreqs`

    Parameters
    ----------
    sr : number > 0 [scalar]
        Audio sampling rate

    n_fft : int > 0 [scalar]
        FFT window size


    Returns
    -------
    freqs : np.ndarray [shape=(1 + n_fft/2,)]
        Frequencies `(0, sr/n_fft, 2*sr/n_fft, ..., sr/2)`


    Examples
    --------
    >> librosa.fft_frequencies(sr=48000, n_fft=16)
    array([     0.   ,   1378.125,   2756.25 ,   4134.375,
             5512.5  ,   6890.625,   8268.75 ,   9646.875,  11025.   ])

    '''

    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_fft//2),
                       endpoint=True)


def cqt_frequencies(n_bins, fmin, bins_per_octave=12, tuning=0.0):
    """Compute the center frequencies of Constant-Q bins.

    Examples
    --------
    >> # Get the CQT frequencies for 24 notes, starting at C2
    >> librosa.cqt_frequencies(24, fmin=librosa.note_to_hz('C2'))
    array([  65.406,   69.296,   73.416,   77.782,   82.407,   87.307,
             92.499,   97.999,  103.826,  110.   ,  116.541,  123.471,
            130.813,  138.591,  146.832,  155.563,  164.814,  174.614,
            184.997,  195.998,  207.652,  220.   ,  233.082,  246.942])

    Parameters
    ----------
    n_bins  : int > 0 [scalar]
        Number of constant-Q bins

    fmin    : float > 0 [scalar]
        Minimum frequency

    bins_per_octave : int > 0 [scalar]
        Number of bins per octave

    tuning : float in `[-0.5, +0.5)`
        Deviation from A440 tuning in fractional bins (cents)

    Returns
    -------
    frequencies : np.ndarray [shape=(n_bins,)]
        Center frequency for each CQT bin
    """

    correction = 2.0**(float(tuning) / bins_per_octave)
    frequencies = 2.0**(np.arange(0, n_bins, dtype=float) / bins_per_octave)

    return correction * fmin * frequencies


def mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0, htk=False):
    """Compute the center frequencies of mel bands.

    Parameters
    ----------
    n_mels    : int > 0 [scalar]
        number of Mel bins

    fmin      : float >= 0 [scalar]
        minimum frequency (Hz)

    fmax      : float >= 0 [scalar]
        maximum frequency (Hz)

    htk       : bool
        use HTK formula instead of Slaney

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_mels,)]
        vector of n_mels frequencies in Hz which are uniformly spaced on the Mel
        axis.

    Examples
    --------
    >> librosa.mel_frequencies(n_mels=40)
    array([     0.   ,     85.317,    170.635,    255.952,
              341.269,    426.586,    511.904,    597.221,
              682.538,    767.855,    853.173,    938.49 ,
             1024.856,   1119.114,   1222.042,   1334.436,
             1457.167,   1591.187,   1737.532,   1897.337,
             2071.84 ,   2262.393,   2470.47 ,   2697.686,
             2945.799,   3216.731,   3512.582,   3835.643,
             4188.417,   4573.636,   4994.285,   5453.621,
             5955.205,   6502.92 ,   7101.009,   7754.107,
             8467.272,   9246.028,  10096.408,  11025.   ])

    """

    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = hz_to_mel(fmin, htk=htk)
    max_mel = hz_to_mel(fmax, htk=htk)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return mel_to_hz(mels, htk=htk)


def tempo_frequencies(n_bins, hop_length=512, sr=48000):
    '''Compute the frequencies (in beats-per-minute) corresponding
    to an onset auto-correlation or tempogram matrix.

    Parameters
    ----------
    n_bins : int > 0
        The number of lag bins

    hop_length : int > 0
        The number of samples between each bin

    sr : number > 0
        The audio sampling rate

    Returns
    -------
    bin_frequencies : ndarray [shape=(n_bins,)]
        vector of bin frequencies measured in BPM.

        .. note:: `bin_frequencies[0] = +np.inf` corresponds to 0-lag

    Examples
    --------
    Get the tempo frequencies corresponding to a 384-bin (8-second) tempogram

    >> librosa.tempo_frequencies(384)
    array([      inf,  2583.984,  1291.992, ...,     6.782,
               6.764,     6.747])
    '''

    bin_frequencies = np.zeros(int(n_bins), dtype=np.float)

    bin_frequencies[0] = np.inf
    bin_frequencies[1:] = 60.0 * sr / (hop_length * np.arange(1.0, n_bins))

    return bin_frequencies


# A-weighting should be capitalized: suppress the naming warning
def A_weighting(frequencies, min_db=-80.0):     # pylint: disable=invalid-name
    '''Compute the A-weighting of a set of frequencies.

    Parameters
    ----------
    frequencies : scalar or np.ndarray [shape=(n,)]
        One or more frequencies (in Hz)

    min_db : float [scalar] or None
        Clip weights below this threshold.
        If `None`, no clipping is performed.

    Returns
    -------
    A_weighting : scalar or np.ndarray [shape=(n,)]
        `A_weighting[i]` is the A-weighting of `frequencies[i]`

    See Also
    --------
    perceptual_weighting


    Examples
    --------

    Get the A-weighting for CQT frequencies

    >> import matplotlib.pyplot as plt
    >> freqs = librosa.cqt_frequencies(108, librosa.note_to_hz('C1'))
    >> aw = librosa.A_weighting(freqs)
    >> plt.plot(freqs, aw)
    >> plt.xlabel('Frequency (Hz)')
    >> plt.ylabel('Weighting (log10)')
    >> plt.title('A-Weighting of CQT frequencies')

    '''

    # Vectorize to make our lives easier
    frequencies = np.atleast_1d(frequencies)

    # Pre-compute squared frequency
    f_sq = frequencies**2.0

    const = np.array([12200, 20.6, 107.7, 737.9])**2.0

    weights = 2.0 + 20.0 * (np.log10(const[0]) + 4 * np.log10(frequencies)
                            - np.log10(f_sq + const[0])
                            - np.log10(f_sq + const[1])
                            - 0.5 * np.log10(f_sq + const[2])
                            - 0.5 * np.log10(f_sq + const[3]))

    if min_db is not None:
        weights = np.maximum(min_db, weights)

    return weights


@cache(level=10)
def get_window(window, Nx, fftbins=True):
    '''Compute a window function.

    This is a wrapper for `scipy.signal.get_window` that additionally
    supports callable or pre-computed windows.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        The window specification:

        - If string, it's the name of the window function (e.g., `'hann'`)
        - If tuple, it's the name of the window function and any parameters
          (e.g., `('kaiser', 4.0)`)
        - If numeric, it is treated as the beta parameter of the `'kaiser'`
          window, as in `scipy.signal.get_window`.
        - If callable, it's a function that accepts one integer argument
          (the window length)
        - If list-like, it's a pre-computed window of the correct length `Nx`

    Nx : int > 0
        The length of the window

    fftbins : bool, optional
        If True (default), create a periodic window for use with FFT
        If False, create a symmetric window for filter design applications.

    Returns
    -------
    get_window : np.ndarray
        A window of length `Nx` and type `window`

    See Also
    --------
    scipy.signal.get_window

    Notes
    -----
    This function caches at level 10.

    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length != `n_fft`,
        or is otherwise mis-specified.
    '''
    if six.callable(window):
        return window(Nx)

    elif (isinstance(window, (six.string_types, tuple)) or
          np.isscalar(window)):

        return sp.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise ParameterError('Window size mismatch: '
                             '{:d} != {:d}'.format(len(window), Nx))
    else:
        raise ParameterError('Invalid window specification: {}'.format(window))
        
        
@cache(level=10)
def window_bandwidth(window, n=1000):
    '''Get the equivalent noise bandwidth of a window function.


    Parameters
    ----------
    window : callable or string
        A window function, or the name of a window function.
        Examples:
        - scipy.signal.hann
        - 'boxcar'

    n : int > 0
        The number of coefficients to use in estimating the
        window bandwidth

    Returns
    -------
    bandwidth : float
        The equivalent noise bandwidth (in FFT bins) of the
        given window function

    Notes
    -----
    This function caches at level 10.

    See Also
    --------
    get_window
    '''

    if hasattr(window, '__name__'):
        key = window.__name__
    else:
        key = window

    if key not in WINDOW_BANDWIDTHS:
        win = get_window(window, n)
        WINDOW_BANDWIDTHS[key] = n * np.sum(win**2) / np.sum(np.abs(win))**2

    return WINDOW_BANDWIDTHS[key]

