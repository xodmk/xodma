# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodmaMiscUtil.py))::__
#
#
# XODMK Audio Tools - Audio Decomposition Miscellaneous Utilities
#
#
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import sys
import six
import numpy as np
import scipy.ndimage
import scipy.sparse


rootDir = '../../'
sys.path.insert(0, rootDir+'audio/xodma')

from cache import cache
import xodmaSpectralUtil as spectralUtil
from xodmaParameterError import ParameterError



__all__ = ['tiny', 'fix_length', 'fix_frames', 'axis_sort', 'normalize',
           'valid_int', 'valid_intervals', 'pad_center', 
           'match_intervals', 'match_events', 'localmax',
           'softmask', 'sparsify_rows', 'index_to_slice', 'sync']



def tiny(x):
    '''Compute the tiny-value corresponding to an input's data type.

    This is the smallest "usable" number representable in `x`'s
    data type (e.g., float32).

    This is primarily useful for determining a threshold for
    numerical underflow in division or multiplication operations.

    Parameters
    ----------
    x : number or np.ndarray
        The array to compute the tiny-value for.
        All that matters here is `x.dtype`.

    Returns
    -------
    tiny_value : float
        The smallest positive usable number for the type of `x`.
        If `x` is integer-typed, then the tiny value for `np.float32`
        is returned instead.

    See Also
    --------
    numpy.finfo

    Examples
    --------

    For a standard double-precision floating point number:

    >>> librosa.util.tiny(1.0)
    2.2250738585072014e-308

    Or explicitly as double-precision

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float64))
    2.2250738585072014e-308

    Or complex numbers

    >>> librosa.util.tiny(1j)
    2.2250738585072014e-308

    Single-precision floating point:

    >>> librosa.util.tiny(np.asarray(1e-5, dtype=np.float32))
    1.1754944e-38

    Integer

    >>> librosa.util.tiny(5)
    1.1754944e-38
    '''

    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.float64) or np.issubdtype(x.dtype, np.complex128):
        dtype = x.dtype
    else:
        dtype = np.float32

    return np.finfo(dtype).tiny


def fix_length(data, size, axis=-1, **kwargs):
    '''Fix the length an array `data` to exactly `size`.

    If `data.shape[axis] < n`, pad according to the provided kwargs.
    By default, `data` is padded with trailing zeros.

    Examples
    --------
    >>> y = np.arange(7)
    >>> # Default: pad with zeros
    >>> librosa.util.fix_length(y, 10)
    array([0, 1, 2, 3, 4, 5, 6, 0, 0, 0])
    >>> # Trim to a desired length
    >>> librosa.util.fix_length(y, 5)
    array([0, 1, 2, 3, 4])
    >>> # Use edge-padding instead of zeros
    >>> librosa.util.fix_length(y, 10, mode='edge')
    array([0, 1, 2, 3, 4, 5, 6, 6, 6, 6])

    Parameters
    ----------
    data : np.ndarray
      array to be length-adjusted

    size : int >= 0 [scalar]
      desired length of the array

    axis : int, <= data.ndim
      axis along which to fix length

    kwargs : additional keyword arguments
        Parameters to `np.pad()`

    Returns
    -------
    data_fixed : np.ndarray [shape=data.shape]
        `data` either trimmed or padded to length `size`
        along the specified axis.

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[slices]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths, **kwargs)

    return data


def fix_frames(frames, x_min=0, x_max=None, pad=True):
    '''Fix a list of frames to lie within [x_min, x_max]

    Examples
    --------
    >>> # Generate a list of frame indices
    >>> frames = np.arange(0, 1000.0, 50)
    >>> frames
    array([   0.,   50.,  100.,  150.,  200.,  250.,  300.,  350.,
            400.,  450.,  500.,  550.,  600.,  650.,  700.,  750.,
            800.,  850.,  900.,  950.])
    >>> # Clip to span at most 250
    >>> librosa.util.fix_frames(frames, x_max=250)
    array([  0,  50, 100, 150, 200, 250])
    >>> # Or pad to span up to 2500
    >>> librosa.util.fix_frames(frames, x_max=2500)
    array([   0,   50,  100,  150,  200,  250,  300,  350,  400,
            450,  500,  550,  600,  650,  700,  750,  800,  850,
            900,  950, 2500])
    >>> librosa.util.fix_frames(frames, x_max=2500, pad=False)
    array([  0,  50, 100, 150, 200, 250, 300, 350, 400, 450, 500,
           550, 600, 650, 700, 750, 800, 850, 900, 950])

    >>> # Or starting away from zero
    >>> frames = np.arange(200, 500, 33)
    >>> frames
    array([200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497])
    >>> librosa.util.fix_frames(frames, x_max=500)
    array([  0, 200, 233, 266, 299, 332, 365, 398, 431, 464, 497,
           500])


    Parameters
    ----------
    frames : np.ndarray [shape=(n_frames,)]
        List of non-negative frame indices

    x_min : int >= 0 or None
        Minimum allowed frame index

    x_max : int >= 0 or None
        Maximum allowed frame index

    pad : boolean
        If `True`, then `frames` is expanded to span the full range
        `[x_min, x_max]`

    Returns
    -------
    fixed_frames : np.ndarray [shape=(n_fixed_frames,), dtype=int]
        Fixed frame indices, flattened and sorted

    Raises
    ------
    ParameterError
        If `frames` contains negative values
    '''

    frames = np.asarray(frames)

    if np.any(frames < 0):
        raise ParameterError('Negative frame index detected')

    if pad and (x_min is not None or x_max is not None):
        frames = np.clip(frames, x_min, x_max)

    if pad:
        pad_data = []
        if x_min is not None:
            pad_data.append(x_min)
        if x_max is not None:
            pad_data.append(x_max)
        frames = np.concatenate((pad_data, frames))

    if x_min is not None:
        frames = frames[frames >= x_min]

    if x_max is not None:
        frames = frames[frames <= x_max]

    return np.unique(frames).astype(int)


def axis_sort(S, axis=-1, index=False, value=None):
    '''Sort an array along its rows or columns.

    Examples
    --------
    Visualize NMF output for a spectrogram S

    >>> # Sort the columns of W by peak frequency bin
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> S = np.abs(librosa.stft(y))
    >>> W, H = librosa.decompose.decompose(S, n_components=32)
    >>> W_sort = librosa.util.axis_sort(W)

    Or sort by the lowest frequency bin

    >>> W_sort = librosa.util.axis_sort(W, value=np.argmin)

    Or sort the rows instead of the columns

    >>> W_sort_rows = librosa.util.axis_sort(W, axis=0)

    Get the sorting index also, and use it to permute the rows of H

    >>> W_sort, idx = librosa.util.axis_sort(W, index=True)
    >>> H_sort = H[idx, :]

    >>> import matplotlib.pyplot as plt
    >>> plt.figure()
    >>> plt.subplot(2, 2, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(W, ref=np.max),
    ...                          y_axis='log')
    >>> plt.title('W')
    >>> plt.subplot(2, 2, 2)
    >>> librosa.display.specshow(H, x_axis='time')
    >>> plt.title('H')
    >>> plt.subplot(2, 2, 3)
    >>> librosa.display.specshow(librosa.amplitude_to_db(W_sort,
    ...                                                  ref=np.max),
    ...                          y_axis='log')
    >>> plt.title('W sorted')
    >>> plt.subplot(2, 2, 4)
    >>> librosa.display.specshow(H_sort, x_axis='time')
    >>> plt.title('H sorted')
    >>> plt.tight_layout()


    Parameters
    ----------
    S : np.ndarray [shape=(d, n)]
        Array to be sorted

    axis : int [scalar]
        The axis along which to compute the sorting values

        - `axis=0` to sort rows by peak column index
        - `axis=1` to sort columns by peak row index

    index : boolean [scalar]
        If true, returns the index array as well as the permuted data.

    value : function
        function to return the index corresponding to the sort order.
        Default: `np.argmax`.

    Returns
    -------
    S_sort : np.ndarray [shape=(d, n)]
        `S` with the columns or rows permuted in sorting order

    idx : np.ndarray (optional) [shape=(d,) or (n,)]
        If `index == True`, the sorting index used to permute `S`.
        Length of `idx` corresponds to the selected `axis`.

    Raises
    ------
    ParameterError
        If `S` does not have exactly 2 dimensions (`S.ndim != 2`)
    '''

    if value is None:
        value = np.argmax

    if S.ndim != 2:
        raise ParameterError('axis_sort is only defined for 2D arrays')

    bin_idx = value(S, axis=np.mod(1-axis, S.ndim))
    idx = np.argsort(bin_idx)

    sort_slice = [slice(None)] * S.ndim
    sort_slice[axis] = idx

    if index:
        return S[sort_slice], idx
    else:
        return S[sort_slice]


def normalize(S, norm=np.inf, axis=0, threshold=None, fill=None):
    '''Normalize an array along a chosen axis.

    Given a norm (described below) and a target axis, the input
    array is scaled so that

        `norm(S, axis=axis) == 1`

    For example, `axis=0` normalizes each column of a 2-d array
    by aggregating over the rows (0-axis).
    Similarly, `axis=1` normalizes each row of a 2-d array.

    This function also supports thresholding small-norm slices:
    any slice (i.e., row or column) with norm below a specified
    `threshold` can be left un-normalized, set to all-zeros, or
    filled with uniform non-zero values that normalize to 1.

    Note: the semantics of this function differ from
    `scipy.linalg.norm` in two ways: multi-dimensional arrays
    are supported, but matrix-norms are not.


    Parameters
    ----------
    S : np.ndarray
        The matrix to normalize

    norm : {np.inf, -np.inf, 0, float > 0, None}
        - `np.inf`  : maximum absolute value
        - `-np.inf` : mininum absolute value
        - `0`    : number of non-zeros (the support)
        - float  : corresponding l_p norm
            See `scipy.linalg.norm` for details.
        - None : no normalization is performed

    axis : int [scalar]
        Axis along which to compute the norm.

    threshold : number > 0 [optional]
        Only the columns (or rows) with norm at least `threshold` are
        normalized.

        By default, the threshold is determined from
        the numerical precision of `S.dtype`.

    fill : None or bool
        If None, then columns (or rows) with norm below `threshold`
        are left as is.

        If False, then columns (rows) with norm below `threshold`
        are set to 0.

        If True, then columns (rows) with norm below `threshold`
        are filled uniformly such that the corresponding norm is 1.

        .. note:: `fill=True` is incompatible with `norm=0` because
            no uniform vector exists with l0 "norm" equal to 1.

    Returns
    -------
    S_norm : np.ndarray [shape=S.shape]
        Normalized array

    Raises
    ------
    ParameterError
        If `norm` is not among the valid types defined above

        If `S` is not finite

        If `fill=True` and `norm=0`

    See Also
    --------
    scipy.linalg.norm

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct an example matrix
    >>> S = np.vander(np.arange(-2.0, 2.0))
    >>> S
    array([[-8.,  4., -2.,  1.],
           [-1.,  1., -1.,  1.],
           [ 0.,  0.,  0.,  1.],
           [ 1.,  1.,  1.,  1.]])
    >>> # Max (l-infinity)-normalize the columns
    >>> librosa.util.normalize(S)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # Max (l-infinity)-normalize the rows
    >>> librosa.util.normalize(S, axis=1)
    array([[-1.   ,  0.5  , -0.25 ,  0.125],
           [-1.   ,  1.   , -1.   ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 1.   ,  1.   ,  1.   ,  1.   ]])
    >>> # l1-normalize the columns
    >>> librosa.util.normalize(S, norm=1)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    >>> # l2-normalize the columns
    >>> librosa.util.normalize(S, norm=2)
    array([[-0.985,  0.943, -0.816,  0.5  ],
           [-0.123,  0.236, -0.408,  0.5  ],
           [ 0.   ,  0.   ,  0.   ,  0.5  ],
           [ 0.123,  0.236,  0.408,  0.5  ]])

    >>> # Thresholding and filling
    >>> S[:, -1] = 1e-308
    >>> S
    array([[ -8.000e+000,   4.000e+000,  -2.000e+000,
              1.000e-308],
           [ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.000e+000,   1.000e+000,   1.000e+000,
              1.000e-308]])

    >>> # By default, small-norm columns are left untouched
    >>> librosa.util.normalize(S)
    array([[ -1.000e+000,   1.000e+000,  -1.000e+000,
              1.000e-308],
           [ -1.250e-001,   2.500e-001,  -5.000e-001,
              1.000e-308],
           [  0.000e+000,   0.000e+000,   0.000e+000,
              1.000e-308],
           [  1.250e-001,   2.500e-001,   5.000e-001,
              1.000e-308]])
    >>> # Small-norm columns can be zeroed out
    >>> librosa.util.normalize(S, fill=False)
    array([[-1.   ,  1.   , -1.   ,  0.   ],
           [-0.125,  0.25 , -0.5  ,  0.   ],
           [ 0.   ,  0.   ,  0.   ,  0.   ],
           [ 0.125,  0.25 ,  0.5  ,  0.   ]])
    >>> # Or set to constant with unit-norm
    >>> librosa.util.normalize(S, fill=True)
    array([[-1.   ,  1.   , -1.   ,  1.   ],
           [-0.125,  0.25 , -0.5  ,  1.   ],
           [ 0.   ,  0.   ,  0.   ,  1.   ],
           [ 0.125,  0.25 ,  0.5  ,  1.   ]])
    >>> # With an l1 norm instead of max-norm
    >>> librosa.util.normalize(S, norm=1, fill=True)
    array([[-0.8  ,  0.667, -0.5  ,  0.25 ],
           [-0.1  ,  0.167, -0.25 ,  0.25 ],
           [ 0.   ,  0.   ,  0.   ,  0.25 ],
           [ 0.1  ,  0.167,  0.25 ,  0.25 ]])
    '''

    # Avoid div-by-zero
    if threshold is None:
        threshold = tiny(S)

    elif threshold <= 0:
        print('ERROR: threshold={} must be strictly '
                             'positive'.format(threshold))
        return False

    if fill not in [None, False, True]:
        print('ERROR: fill={} must be None or boolean'.format(fill))
        return False

    if not np.all(np.isfinite(S)):
        print('ERROR: Input must be finite')
        return False

    # All norms only depend on magnitude, let's do that first
    mag = np.abs(S).astype(np.float)

    # For max/min norms, filling with 1 works
    fill_norm = 1

    if norm == np.inf:
        length = np.max(mag, axis=axis, keepdims=True)

    elif norm == -np.inf:
        length = np.min(mag, axis=axis, keepdims=True)

    elif norm == 0:
        if fill is True:
            print('ERROR: Cannot normalize with norm=0 and fill=True')
            return False

        length = np.sum(mag > 0, axis=axis, keepdims=True, dtype=mag.dtype)

    elif np.issubdtype(type(norm), np.number) and norm > 0:
        length = np.sum(mag**norm, axis=axis, keepdims=True)**(1./norm)

        if axis is None:
            fill_norm = mag.size**(-1./norm)
        else:
            fill_norm = mag.shape[axis]**(-1./norm)

    elif norm is None:
        return S

    else:
        print('ERROR: Unsupported norm: {}'.format(repr(norm)))
        return False

    # indices where norm is below the threshold
    small_idx = length < threshold

    if fill is None:
        # Leave small indices un-normalized
        length[small_idx] = 1.0
        Snorm = S / length

    elif fill:
        # If we have a non-zero fill value, we locate those entries by
        # doing a nan-divide.
        # If S was finite, then length is finite (except for small positions)
        length[small_idx] = np.nan
        Snorm = S / length
        Snorm[np.isnan(Snorm)] = fill_norm
    else:
        # Set small values to zero by doing an inf-divide.
        # This is safe (by IEEE-754) as long as S is finite.
        length[small_idx] = np.inf
        Snorm = S / length

    return Snorm


#def normalize(y):
#    '''Normalize audio to +/- 1.0
#    
#    '''
#    
#        # All norms only depend on magnitude, let's do that first
#    mag = np.abs(y).astype(np.float)
#    
#    ynorm = y*mag**(-1)
#
#    return ynorm


def valid_int(x, cast=None):
    '''Ensure that an input value is integer-typed.
    This is primarily useful for ensuring integrable-valued
    array indices.

    Parameters
    ----------
    x : number
        A scalar value to be cast to int

    cast : function [optional]
        A function to modify `x` before casting.
        Default: `np.floor`

    Returns
    -------
    x_int : int
        `x_int = int(cast(x))`

    Raises
    ------
    ParameterError
        If `cast` is provided and is not callable.
    '''

    if cast is None:
        cast = np.floor

    if not six.callable(cast):
        raise ParameterError('cast parameter must be callable')

    return int(cast(x))


def valid_intervals(intervals):
    '''Ensure that an array is a valid representation of time intervals:

        - intervals.ndim == 2
        - intervals.shape[1] == 2

    Parameters
    ----------
    intervals : np.ndarray [shape=(n, 2)]
        set of time intervals

    Returns
    -------
    valid : bool
        True if `intervals` passes validation.
    '''

    if intervals.ndim != 2 or intervals.shape[-1] != 2:
        raise ParameterError('intervals must have shape (n, 2)')

    return True


def pad_center(data, size, axis=-1, **kwargs):
    '''Wrapper for np.pad to automatically center an array prior to padding.
    This is analogous to `str.center()`

    Examples
    --------
    >>> # Generate a vector
    >>> data = np.ones(5)
    >>> librosa.util.pad_center(data, 10, mode='constant')
    array([ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.])

    >>> # Pad a matrix along its first dimension
    >>> data = np.ones((3, 5))
    >>> librosa.util.pad_center(data, 7, axis=0)
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    >>> # Or its second dimension
    >>> librosa.util.pad_center(data, 7, axis=1)
    array([[ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.],
           [ 0.,  1.,  1.,  1.,  1.,  1.,  0.]])

    Parameters
    ----------
    data : np.ndarray
        Vector to be padded and centered

    size : int >= len(data) [scalar]
        Length to pad `data`

    axis : int
        Axis along which to pad and center the data

    kwargs : additional keyword arguments
      arguments passed to `np.pad()`

    Returns
    -------
    data_padded : np.ndarray
        `data` centered and padded to length `size` along the
        specified axis

    Raises
    ------
    ParameterError
        If `size < data.shape[axis]`

    See Also
    --------
    numpy.pad
    '''

    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be '
                              'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)


def match_intervals(intervals_from, intervals_to):
    '''Match one set of time intervals to another.

    This can be useful for tasks such as mapping beat timings
    to segments.

    .. note:: A target interval may be matched to multiple source
      intervals.

    Parameters
    ----------
    intervals_from : np.ndarray [shape=(n, 2)]
        The time range for source intervals.
        The `i` th interval spans time `intervals_from[i, 0]`
        to `intervals_from[i, 1]`.
        `intervals_from[0, 0]` should be 0, `intervals_from[-1, 1]`
        should be the track duration.

    intervals_to : np.ndarray [shape=(m, 2)]
        Analogous to `intervals_from`.

    Returns
    -------
    interval_mapping : np.ndarray [shape=(n,)]
        For each interval in `intervals_from`, the
        corresponding interval in `intervals_to`.

    See Also
    --------
    match_events

    Raises
    ------
    ParameterError
        If either array of input intervals is not the correct shape
    '''

    if len(intervals_from) == 0 or len(intervals_to) == 0:
        raise ParameterError('Attempting to match empty interval list')

    # Verify that the input intervals has correct shape and size
    valid_intervals(intervals_from)
    valid_intervals(intervals_to)

    # The overlap score of a beat with a segment is defined as
    #   max(0, min(beat_end, segment_end) - max(beat_start, segment_start))
    output = np.empty(len(intervals_from), dtype=np.int)

    n_rows = int(spectralUtil.MAX_MEM_BLOCK / (len(intervals_to) * intervals_to.itemsize))
    n_rows = max(1, n_rows)

    for bl_s in range(0, len(intervals_from), n_rows):
        bl_t = min(bl_s + n_rows, len(intervals_from))
        tmp_from = intervals_from[bl_s:bl_t]

        starts = np.maximum.outer(tmp_from[:, 0], intervals_to[:, 0])
        ends = np.minimum.outer(tmp_from[:, 1], intervals_to[:, 1])
        score = np.maximum(0, ends - starts)

        output[bl_s:bl_t] = np.argmax(score, axis=-1)

    return output


def match_events(events_from, events_to, left=True, right=True):
    '''Match one set of events to another.

    This is useful for tasks such as matching beats to the nearest
    detected onsets, or frame-aligned events to the nearest zero-crossing.

    .. note:: A target event may be matched to multiple source events.

    Examples
    --------
    >>> # Sources are multiples of 7
    >>> s_from = np.arange(0, 100, 7)
    >>> s_from
    array([ 0,  7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91,
           98])
    >>> # Targets are multiples of 10
    >>> s_to = np.arange(0, 100, 10)
    >>> s_to
    array([ 0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> # Find the matching
    >>> idx = librosa.util.match_events(s_from, s_to)
    >>> idx
    array([0, 1, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 8, 9, 9])
    >>> # Print each source value to its matching target
    >>> zip(s_from, s_to[idx])
    [(0, 0), (7, 10), (14, 10), (21, 20), (28, 30), (35, 30),
     (42, 40), (49, 50), (56, 60), (63, 60), (70, 70), (77, 80),
     (84, 80), (91, 90), (98, 90)]

    Parameters
    ----------
    events_from : ndarray [shape=(n,)]
      Array of events (eg, times, sample or frame indices) to match from.

    events_to : ndarray [shape=(m,)]
      Array of events (eg, times, sample or frame indices) to
      match against.

    left : bool
    right : bool
        If `False`, then matched events cannot be to the left (or right)
        of source events.

    Returns
    -------
    event_mapping : np.ndarray [shape=(n,)]
        For each event in `events_from`, the corresponding event
        index in `events_to`.

        `event_mapping[i] == arg min |events_from[i] - events_to[:]|`

    See Also
    --------
    match_intervals

    Raises
    ------
    ParameterError
        If either array of input events is not the correct shape
    '''

    if len(events_from) == 0 or len(events_to) == 0:
        raise ParameterError('Attempting to match empty event list')

    # If we can't match left or right, then only strict equivalence
    # counts as a match.
    if not (left or right) and not np.all(np.in1d(events_from, events_to)):
            raise ParameterError('Cannot match events with left=right=False '
                                 'and events_from is not contained '
                                 'in events_to')

    # If we can't match to the left, then there should be at least one
    # target event greater-equal to every source event
    if (not left) and max(events_to) < max(events_from):
        raise ParameterError('Cannot match events with left=False '
                             'and max(events_to) < max(events_from)')

    # If we can't match to the right, then there should be at least one
    # target event less-equal to every source event
    if (not right) and min(events_to) > min(events_from):
        raise ParameterError('Cannot match events with right=False '
                             'and min(events_to) > min(events_from)')

    # Pre-allocate the output array
    output = np.empty_like(events_from, dtype=np.int)

    # Compute how many rows we can process at once within the memory block
    n_rows = int(spectralUtil.MAX_MEM_BLOCK / (np.prod(output.shape[1:]) 
                     * len(events_to) * events_from.itemsize))

    # Make sure we can at least make some progress
    n_rows = max(1, n_rows)

    # Iterate over blocks of the data
    for bl_s in range(0, len(events_from), n_rows):
        bl_t = min(bl_s + n_rows, len(events_from))

        event_block = events_from[bl_s:bl_t]

        # distance[i, j] = |events_from - events_to[j]|
        distance = np.abs(np.subtract.outer(event_block,
                                            events_to)).astype(np.float)

        # If we can't match to the right, squash all comparisons where
        # events_to[j] > events_from[i]
        if not right:
            distance[np.less.outer(event_block, events_to)] = np.nan

        # If we can't match to the left, squash all comparisons where
        # events_to[j] < events_from[i]
        if not left:
            distance[np.greater.outer(event_block, events_to)] = np.nan

        # Find the minimum distance point from whatever's left after squashing
        output[bl_s:bl_t] = np.nanargmin(distance, axis=-1)

    return output


def localmax(x, axis=0):
    """Find local maxima in an array `x`.

    Examples
    --------
    >>> x = np.array([1, 0, 1, 2, -1, 0, -2, 1])
    >>> librosa.util.localmax(x)
    array([False, False, False,  True, False,  True, False,  True], dtype=bool)

    >>> # Two-dimensional example
    >>> x = np.array([[1,0,1], [2, -1, 0], [2, 1, 3]])
    >>> librosa.util.localmax(x, axis=0)
    array([[False, False, False],
           [ True, False, False],
           [False,  True,  True]], dtype=bool)
    >>> librosa.util.localmax(x, axis=1)
    array([[False, False,  True],
           [False, False,  True],
           [False, False,  True]], dtype=bool)

    Parameters
    ----------
    x     : np.ndarray [shape=(d1,d2,...)]
      input vector or array

    axis : int
      axis along which to compute local maximality

    Returns
    -------
    m     : np.ndarray [shape=x.shape, dtype=bool]
        indicator array of local maximality along `axis`

    """

    paddings = [(0, 0)] * x.ndim
    paddings[axis] = (1, 1)

    x_pad = np.pad(x, paddings, mode='edge')

    inds1 = [slice(None)] * x.ndim
    inds1[axis] = slice(0, -2)

    inds2 = [slice(None)] * x.ndim
    inds2[axis] = slice(2, x_pad.shape[axis])

    return (x > x_pad[inds1]) & (x >= x_pad[inds2])


def softmask(X, X_ref, power=1, split_zeros=False):
    '''Robustly compute a softmask operation.

        `M = X**power / (X**power + X_ref**power)`


    Parameters
    ----------
    X : np.ndarray
        The (non-negative) input array corresponding to the positive mask elements

    X_ref : np.ndarray
        The (non-negative) array of reference or background elements.
        Must have the same shape as `X`.

    power : number > 0 or np.inf
        If finite, returns the soft mask computed in a numerically stable way

        If infinite, returns a hard (binary) mask equivalent to `X > X_ref`.
        Note: for hard masks, ties are always broken in favor of `X_ref` (`mask=0`).


    split_zeros : bool
        If `True`, entries where `X` and X`_ref` are both small (close to 0)
        will receive mask values of 0.5.

        Otherwise, the mask is set to 0 for these entries.


    Returns
    -------
    mask : np.ndarray, shape=`X.shape`
        The output mask array

    Raises
    ------
    ParameterError
        If `X` and `X_ref` have different shapes.

        If `X` or `X_ref` are negative anywhere

        If `power <= 0`

    Examples
    --------

    >>> X = 2 * np.ones((3, 3))
    >>> X_ref = np.vander(np.arange(3.0))
    >>> X
    array([[ 2.,  2.,  2.],
           [ 2.,  2.,  2.],
           [ 2.,  2.,  2.]])
    >>> X_ref
    array([[ 0.,  0.,  1.],
           [ 1.,  1.,  1.],
           [ 4.,  2.,  1.]])
    >>> librosa.util.softmask(X, X_ref, power=1)
    array([[ 1.   ,  1.   ,  0.667],
           [ 0.667,  0.667,  0.667],
           [ 0.333,  0.5  ,  0.667]])
    >>> librosa.util.softmask(X_ref, X, power=1)
    array([[ 0.   ,  0.   ,  0.333],
           [ 0.333,  0.333,  0.333],
           [ 0.667,  0.5  ,  0.333]])
    >>> librosa.util.softmask(X, X_ref, power=2)
    array([[ 1. ,  1. ,  0.8],
           [ 0.8,  0.8,  0.8],
           [ 0.2,  0.5,  0.8]])
    >>> librosa.util.softmask(X, X_ref, power=4)
    array([[ 1.   ,  1.   ,  0.941],
           [ 0.941,  0.941,  0.941],
           [ 0.059,  0.5  ,  0.941]])
    >>> librosa.util.softmask(X, X_ref, power=100)
    array([[  1.000e+00,   1.000e+00,   1.000e+00],
           [  1.000e+00,   1.000e+00,   1.000e+00],
           [  7.889e-31,   5.000e-01,   1.000e+00]])
    >>> librosa.util.softmask(X, X_ref, power=np.inf)
    array([[ True,  True,  True],
           [ True,  True,  True],
           [False, False,  True]], dtype=bool)
    '''
    if X.shape != X_ref.shape:
        raise ParameterError('Shape mismatch: {}!={}'.format(X.shape,
                                                             X_ref.shape))

    if np.any(X < 0) or np.any(X_ref < 0):
        raise ParameterError('X and X_ref must be non-negative')

    if power <= 0:
        raise ParameterError('power must be strictly positive')

    # We're working with ints, cast to float.
    dtype = X.dtype
    if not np.issubdtype(dtype, float):
        dtype = np.float32

    # Re-scale the input arrays relative to the larger value
    Z = np.maximum(X, X_ref).astype(dtype)
    bad_idx = (Z < np.finfo(dtype).tiny)
    Z[bad_idx] = 1

    # For finite power, compute the softmask
    if np.isfinite(power):
        mask = (X / Z)**power
        ref_mask = (X_ref / Z)**power
        good_idx = ~bad_idx
        mask[good_idx] /= mask[good_idx] + ref_mask[good_idx]
        # Wherever energy is below energy in both inputs, split the mask
        if split_zeros:
            mask[bad_idx] = 0.5
        else:
            mask[bad_idx] = 0.0
    else:
        # Otherwise, compute the hard mask
        mask = X > X_ref

    return mask





@cache(level=40)
def sparsify_rows(x, quantile=0.01):
    '''
    Return a row-sparse matrix approximating the input `x`.

    Parameters
    ----------
    x : np.ndarray [ndim <= 2]
        The input matrix to sparsify.

    quantile : float in [0, 1.0)
        Percentage of magnitude to discard in each row of `x`

    Returns
    -------
    x_sparse : `scipy.sparse.csr_matrix` [shape=x.shape]
        Row-sparsified approximation of `x`

        If `x.ndim == 1`, then `x` is interpreted as a row vector,
        and `x_sparse.shape == (1, len(x))`.

    Raises
    ------
    ParameterError
        If `x.ndim > 2`

        If `quantile` lies outside `[0, 1.0)`

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    >>> # Construct a Hann window to sparsify
    >>> x = scipy.signal.hann(32)
    >>> x
    array([ 0.   ,  0.01 ,  0.041,  0.09 ,  0.156,  0.236,  0.326,
            0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
            0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
            0.72 ,  0.625,  0.525,  0.424,  0.326,  0.236,  0.156,
            0.09 ,  0.041,  0.01 ,  0.   ])
    >>> # Discard the bottom percentile
    >>> x_sparse = librosa.util.sparsify_rows(x, quantile=0.01)
    >>> x_sparse
    <1x32 sparse matrix of type '<type 'numpy.float64'>'
        with 26 stored elements in Compressed Sparse Row format>
    >>> x_sparse.todense()
    matrix([[ 0.   ,  0.   ,  0.   ,  0.09 ,  0.156,  0.236,  0.326,
              0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
              0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
              0.72 ,  0.625,  0.525,  0.424,  0.326,  0.236,  0.156,
              0.09 ,  0.   ,  0.   ,  0.   ]])
    >>> # Discard up to the bottom 10th percentile
    >>> x_sparse = librosa.util.sparsify_rows(x, quantile=0.1)
    >>> x_sparse
    <1x32 sparse matrix of type '<type 'numpy.float64'>'
        with 20 stored elements in Compressed Sparse Row format>
    >>> x_sparse.todense()
    matrix([[ 0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.   ,  0.326,
              0.424,  0.525,  0.625,  0.72 ,  0.806,  0.879,  0.937,
              0.977,  0.997,  0.997,  0.977,  0.937,  0.879,  0.806,
              0.72 ,  0.625,  0.525,  0.424,  0.326,  0.   ,  0.   ,
              0.   ,  0.   ,  0.   ,  0.   ]])
    '''

    if x.ndim == 1:
        x = x.reshape((1, -1))

    elif x.ndim > 2:
        raise ParameterError('Input must have 2 or fewer dimensions. '
                             'Provided x.shape={}.'.format(x.shape))

    if not 0.0 <= quantile < 1:
        raise ParameterError('Invalid quantile {:.2f}'.format(quantile))

    x_sparse = scipy.sparse.lil_matrix(x.shape, dtype=x.dtype)

    mags = np.abs(x)
    norms = np.sum(mags, axis=1, keepdims=True)

    mag_sort = np.sort(mags, axis=1)
    cumulative_mag = np.cumsum(mag_sort / norms, axis=1)

    threshold_idx = np.argmin(cumulative_mag < quantile, axis=1)

    for i, j in enumerate(threshold_idx):
        idx = np.where(mags[i] >= mag_sort[i, j])
        x_sparse[i, idx] = x[i, idx]

    return x_sparse.tocsr()


def roll_sparse(x, shift, axis=0):
    '''Sparse matrix roll

    This operation is equivalent to ``numpy.roll``, but operates on sparse matrices.

    Parameters
    ----------
    x : scipy.sparse.spmatrix or np.ndarray
        The sparse matrix input

    shift : int
        The number of positions to roll the specified axis

    axis : (0, 1, -1)
        The axis along which to roll.

    Returns
    -------
    x_rolled : same type as `x`
        The rolled matrix, with the same format as `x`

    See Also
    --------
    numpy.roll

    Examples
    --------
    >>> # Generate a random sparse binary matrix
    >>> X = scipy.sparse.lil_matrix(np.random.randint(0, 2, size=(5,5)))
    >>> X_roll = roll_sparse(X, 2, axis=0)  # Roll by 2 on the first axis
    >>> X_dense_r = roll_sparse(X.toarray(), 2, axis=0)  # Equivalent dense roll
    >>> np.allclose(X_roll, X_dense_r.toarray())
    True
    '''
    if not scipy.sparse.isspmatrix(x):
        return np.roll(x, shift, axis=axis)

    # shift-mod-length lets us have shift > x.shape[axis]
    if axis not in [0, 1, -1]:
        raise ParameterError('axis must be one of (0, 1, -1)')

    shift = np.mod(shift, x.shape[axis])

    if shift == 0:
        return x.copy()

    fmt = x.format
    if axis == 0:
        x = x.tocsc()
    elif axis in (-1, 1):
        x = x.tocsr()

    # lil matrix to start
    x_r = scipy.sparse.lil_matrix(x.shape, dtype=x.dtype)

    idx_in = [slice(None)] * x.ndim
    idx_out = [slice(None)] * x_r.ndim

    idx_in[axis] = slice(0, -shift)
    idx_out[axis] = slice(shift, None)
    x_r[tuple(idx_out)] = x[tuple(idx_in)]

    idx_out[axis] = slice(0, shift)
    idx_in[axis] = slice(-shift, None)
    x_r[tuple(idx_out)] = x[tuple(idx_in)]

    return x_r.asformat(fmt)


def index_to_slice(idx, idx_min=None, idx_max=None, step=None, pad=True):
    '''Generate a slice array from an index array.

    Parameters
    ----------
    idx : list-like
        Array of index boundaries

    idx_min : None or int
    idx_max : None or int
        Minimum and maximum allowed indices

    step : None or int
        Step size for each slice.  If `None`, then the default
        step of 1 is used.

    pad : boolean
        If `True`, pad `idx` to span the range `idx_min:idx_max`.

    Returns
    -------
    slices : list of slice
        ``slices[i] = slice(idx[i], idx[i+1], step)``
        Additional slice objects may be added at the beginning or end,
        depending on whether ``pad==True`` and the supplied values for
        `idx_min` and `idx_max`.

    See Also
    --------
    fix_frames

    Examples
    --------
    >>> # Generate slices from spaced indices
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15))
    [slice(20, 35, None), slice(35, 50, None), slice(50, 65, None), slice(65, 80, None),
     slice(80, 95, None)]
    >>> # Pad to span the range (0, 100)
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100)
    [slice(0, 20, None), slice(20, 35, None), slice(35, 50, None), slice(50, 65, None),
     slice(65, 80, None), slice(80, 95, None), slice(95, 100, None)]
    >>> # Use a step of 5 for each slice
    >>> librosa.util.index_to_slice(np.arange(20, 100, 15),
    ...                             idx_min=0, idx_max=100, step=5)
    [slice(0, 20, 5), slice(20, 35, 5), slice(35, 50, 5), slice(50, 65, 5), slice(65, 80, 5),
     slice(80, 95, 5), slice(95, 100, 5)]
    '''

    # First, normalize the index set
    idx_fixed = fix_frames(idx, idx_min, idx_max, pad=pad)

    # Now convert the indices to slices
    return [slice(start, end, step) for (start, end) in zip(idx_fixed, idx_fixed[1:])]


@cache(level=40)
def sync(data, idx, aggregate=None, pad=True, axis=-1):
    """Synchronous aggregation of a multi-dimensional array between boundaries

    .. note::
        In order to ensure total coverage, boundary points may be added
        to `idx`.

        If synchronizing a feature matrix against beat tracker output, ensure
        that frame index numbers are properly aligned and use the same hop length.

    Parameters
    ----------
    data      : np.ndarray
        multi-dimensional array of features

    idx : iterable of ints or slices
        Either an ordered array of boundary indices, or
        an iterable collection of slice objects.


    aggregate : function
        aggregation function (default: `np.mean`)

    pad : boolean
        If `True`, `idx` is padded to span the full range `[0, data.shape[axis]]`

    axis : int
        The axis along which to aggregate data

    Returns
    -------
    data_sync : ndarray
        `data_sync` will have the same dimension as `data`, except that the `axis`
        coordinate will be reduced according to `idx`.

        For example, a 2-dimensional `data` with `axis=-1` should satisfy

        `data_sync[:, i] = aggregate(data[:, idx[i-1]:idx[i]], axis=-1)`

    Raises
    ------
    ParameterError
        If the index set is not of consistent type (all slices or all integers)

    Notes
    -----
    This function caches at level 40.

    Examples
    --------
    Beat-synchronous CQT spectra

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    >>> C = np.abs(librosa.cqt(y=y, sr=sr))
    >>> beats = librosa.util.fix_frames(beats, x_max=C.shape[1])

    By default, use mean aggregation

    >>> C_avg = librosa.util.sync(C, beats)

    Use median-aggregation instead of mean

    >>> C_med = librosa.util.sync(C, beats,
    ...                             aggregate=np.median)

    Or sub-beat synchronization

    >>> sub_beats = librosa.segment.subsegment(C, beats)
    >>> sub_beats = librosa.util.fix_frames(sub_beats, x_max=C.shape[1])
    >>> C_med_sub = librosa.util.sync(C, sub_beats, aggregate=np.median)


    Plot the results

    >>> import matplotlib.pyplot as plt
    >>> beat_t = librosa.frames_to_time(beats, sr=sr)
    >>> subbeat_t = librosa.frames_to_time(sub_beats, sr=sr)
    >>> plt.figure()
    >>> plt.subplot(3, 1, 1)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C,
    ...                                                  ref=np.max),
    ...                          x_axis='time')
    >>> plt.title('CQT power, shape={}'.format(C.shape))
    >>> plt.subplot(3, 1, 2)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C_med,
    ...                                                  ref=np.max),
    ...                          x_coords=beat_t, x_axis='time')
    >>> plt.title('Beat synchronous CQT power, '
    ...           'shape={}'.format(C_med.shape))
    >>> plt.subplot(3, 1, 3)
    >>> librosa.display.specshow(librosa.amplitude_to_db(C_med_sub,
    ...                                                  ref=np.max),
    ...                          x_coords=subbeat_t, x_axis='time')
    >>> plt.title('Sub-beat synchronous CQT power, '
    ...           'shape={}'.format(C_med_sub.shape))
    >>> plt.tight_layout()
    >>> plt.show()

    """

    if aggregate is None:
        aggregate = np.mean

    shape = list(data.shape)

    if np.all([isinstance(_, slice) for _ in idx]):
        slices = idx
    elif np.all([np.issubdtype(type(_), np.integer) for _ in idx]):
        slices = index_to_slice(np.asarray(idx), 0, shape[axis], pad=pad)
    else:
        raise ParameterError('Invalid index set: {}'.format(idx))

    agg_shape = list(shape)
    agg_shape[axis] = len(slices)

    data_agg = np.empty(agg_shape, order='F' if np.isfortran(data) else 'C', dtype=data.dtype)

    idx_in = [slice(None)] * data.ndim
    idx_agg = [slice(None)] * data_agg.ndim

    for (i, segment) in enumerate(slices):
        idx_in[axis] = segment
        idx_agg[axis] = i
        data_agg[tuple(idx_agg)] = aggregate(data[tuple(idx_in)], axis=axis)

    return data_agg


# /////////////////////////////////////////////////////////////////////////////
# misc utils-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //

# Exception classes for librosa


#class odmkSpectralToolsError(Exception):
#    '''The root exception class'''
#    pass
#
#
#class ParameterError(odmkSpectralToolsError):
#    '''Exception class for input parameter errors'''
#    pass
