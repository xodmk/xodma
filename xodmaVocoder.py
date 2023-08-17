# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodmaVocoder.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
#
# XODMK Audio Tools - Phase Vocoder Funk
#
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import numpy as np
# import scipy as sp


currentDir = os.getcwd()
rootDir = os.path.dirname(currentDir)
sys.path.insert(0, rootDir+'/xodma/')

from xodmaAudioTools import resample
from xodmaAudioTools import samples_to_time, time_to_samples, fix_length
from xodmaSpectralTools import amplitude_to_db, stft, istft, magphase, peak_pick
from xodmaMiscUtil import valid_audio

# temp python debugger - use >>>pdb.set_trace() to set break
#import pdb


# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //
    
def pv1(D, rate, hop_length=None):
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`

    Based on the implementation provided by [1]_.

    .. [1] Ellis, D. P. W. "A phase vocoder in Matlab."
        Columbia University, 2002.
        http://www.ee.columbia.edu/~dpwe/resources/matlab/pvoc/

    Examples
    --------
    # >>> # Play at double speed
    # >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    # >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    # >>> D_fast  = librosa.phase_vocoder(D, 2.0, hop_length=512)
    # >>> y_fast  = librosa.istft(D_fast, hop_length=512)
    #
    # >>> # Or play at 1/3 speed
    # >>> y, sr   = librosa.load(librosa.util.example_audio_file())
    # >>> D       = librosa.stft(y, n_fft=2048, hop_length=512)
    # >>> D_slow  = librosa.phase_vocoder(D, 1./3, hop_length=512)
    # >>> y_slow  = librosa.istft(D_slow, hop_length=512)

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

    time_steps = np.arange(0, D.shape[1], rate, dtype=float)

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


def pvTimeStretch(y, rate, n_fft):
    """Time-stretch an audio series by a fixed rate.

    __Parameters:
    :param y: np.ndarray [shape=(n,)] - audio time series
    :param rate: float [scalar > 0] => Time-Stretch -> rate > 1 ; Time-Compress -> rate < 1
    :param n_fft: FFT Length

    __Returns:
    y_stretch : np.ndarray [shape=(rate * n,)]
        audio time series stretched by the specified rate

    __Examples:
    # >>> # Time-Compress: Play at 1/2x speed
    # >>> rate = timeExpand = 0.5
    # >>> vxmod = 0.05     # Vox Modulation Depth - range[0.01, 1.0]
    # >>> yResult = pvTimeStretch(y, rate, n_fft)

    """

    if rate <= 0:
        print('\nrate must be a positive number')
        return

    # Construct the stft
    ySTFT = stft(y, n_fft=n_fft)

    # Stretch by phase vocoding
    stftStretch = pv1(ySTFT, rate)

    # Invert the stft
    yStretch = istft(stftStretch, dtype=y.dtype)

    return yStretch


def pvPitchShift(y, sr, n_fft, n_steps, bins_per_octave=12):
    """ Pitch-shift the waveform by `n_steps` half-steps.

    Parameters
    ----------
    y : np.ndarray [shape=(n,)]
        audio time-series

    sr : number > 0 [scalar]
        audio sampling rate of `y`

    n_fft : FFT Length

    n_steps : float [scalar]
        how many (fractional) half-steps to shift `y`

    bins_per_octave : float > 0 [scalar]
        how many steps per octave


    Returns
    -------
    y_shift : np.ndarray [shape=(n,)]
        The pitch-shifted audio time-series


    See Also
    --------
    time_stretch : time stretching
    librosa.core.phase_vocoder : spectrogram phase vocoder


    Examples
    --------
    Shift up by a major third (four half-steps)

    # >>> y, sr = librosa.load(librosa.util.example_audio_file())
    # >>> y_third = librosa.effects.pitch_shift(y, sr, n_steps=4)

    Shift down by a tritone (six half-steps)

    # >>> y_tritone = librosa.effects.pitch_shift(y, sr, n_steps=-6)

    Shift up by 3 quarter-tones

    # >>> y_three_qt = librosa.effects.pitch_shift(y, sr, n_steps=3,
    ...                                          bins_per_octave=24)
    """

    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), int):
        sys.exit('ERROR: func pitch_shift - bins_per_octave must be a positive integer.')

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # Stretch in time, then resample
    y_shift = resample(pvTimeStretch(y, rate, n_fft), float(sr) / rate, sr)

    # Crop to the same dimension as the input
    return fix_length(y_shift, len(y))


# // *---------------------------------------------------------------------* //

def pvRobotSmith(ySTFT, rate, voxmod, hop_length=None):
    """ XODMK Phase vocoder.  Given an STFT matrix ySTFT, speed up by a factor of `rate`

    __parameters__
    :param ySTFT       : np.ndarray [shape=(d, t), dtype=complex] => STFT output matrix
    :param rate        : float [scalar > 0] => Time-Stretch -> rate > 1 ; Time-Compress -> rate < 1
    :param voxmod      : float 0.001 < x < ? => vox modulation EFFX
    :param hop_length  : int > 0 [scalar] or None
        The number of samples between successive columns of `D`.
        If None, defaults to `n_fft/4 = (D.shape[0]-1)/2`

    __Returns__
    yVocode     : np.ndarray [shape=(d, t / rate), dtype=complex] => phase vocoded STFT

    __Examples__
    # >>> # Time-Compress: Play at 1/2x speed
    # >>> rate = timeExpand = 0.5
    # >>> vxmod = 0.05     # Vox Modulation Depth - range[0.01, 1.0]
    # >>> ySTFT = stft(y, n_fft)
    # >>> yVocodeOut = pvRobotSmith(ySTFT, rate, vxmod)
    # >>> yNewResult = istft(yVocodeOut, dtype=y.dtype)

    """

    n_fft = 2 * (ySTFT.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, ySTFT.shape[1], rate, dtype=float)

    # Create an empty output array
    yVocode = np.zeros((ySTFT.shape[0], len(time_steps)), ySTFT.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, ySTFT.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(ySTFT[:, 0])

    # Pad 0 columns to simplify boundary logic
    ySTFT = np.pad(ySTFT, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):
        columns = ySTFT[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = ((1.0 - alpha) * np.abs(columns[:, 0])
               + alpha * np.abs(columns[:, 1]))

        # Store to output array
        # yVocode[:, t] = mag * np.exp(1.j * phase_acc * voxmod)
        yVocode[:, t] = mag * np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        # phase_acc += phi_advance + dphase
        phase_acc += phi_advance

    return yVocode


def pvRobotStretch(y, rate, vxmod, n_fft):
    """ Phase-Vocoder RobotSmith Time-Stretch an audio series by a fixed rate

    __parameters__
    y       : np.ndarray [shape=(n,)] => audio time series
    rate    : float [scalar > 0] => Time-Stretch -> rate > 1 ; Time-Compress -> rate < 1
    vxmod   : vox modulation EFFX
    n_fft       : FFT Length

    __return__
    yStretch : np.ndarray [shape=(rate * n,)] => modified audio time series


    __examples__

    >> Process Stereo .wav file...
    >> from xodmaAudioTools import load_wav, write_wav
    >> audioSrc = audioSrcDir + '/' + srcNm
    >> [aSrc, aNumChannels, afs, aLength, aSamples] = load_wav(audioSrc, wavLength, True)
    >> aSrc_ch1 = aSrc[:, 0]
    >> aSrc_ch2 = aSrc[:, 1]

    ** vox modulation exp...
    >> vxmod = 0.05     # Vox Modulation Depth - range[0.01, 1.0] :
    >> vxtilt = -0.5    # Vox Mod Stereo
    >> vxmodL = vxmod - (vxtilt * vxmod)
    >> vxmodR = vxmod + (vxtilt * vxmod)

    ** Time-Compress to be twice as fast:  rate = timeCompress = 2.0
    >> yRS_Compress_ch1 = pvRobotStretch(aSrc_ch1, timeCompress, vxmodL)
    >> yRS_Compress_ch2 = pvRobotStretch(aSrc_ch2, timeCompress, vxmodR)
    >> yRS_Compress = np.transpose(np.column_stack((yRS_Compress_ch1, yRS_Compress_ch2)))

    ** Time-Expand to half the original speed: rate = timeExpand = 0.5
    >> yRS_Expand_ch1 = pvRobotStretch(aSrc_ch1, timeExpand, vxmodL)
    >> yRS_Expand_ch2 = pvRobotStretch(aSrc_ch2, timeExpand, vxmodR)
    >> yRS_Expand = np.transpose(np.column_stack((yRS_Expand_ch1, yRS_Expand_ch2)))

    """

    # Check if y is valid nd.array audio time series
    if not valid_audio(y):
        print('\ny must be a nd.array audio time series')
        return

    if rate <= 0:
        print('\nrate must be a positive number')
        return

    ySTFT = stft(y, n_fft=n_fft)
    stftStretch = pvRobotSmith(ySTFT, rate, vxmod)
    yStretch = istft(stftStretch, dtype=y.dtype)

    return yStretch


# // *---------------------------------------------------------------------* //

def pvPeakHarmonics(ySTFT, rate, numPeaks, hop_length=None):
    """ XODMK Phase vocoder.  Given an STFT matrix ySTFT, speed up by a factor of `rate`

    __parameters__
    ySTFT       : np.ndarray [shape=(d, t), dtype=complex] => STFT output matrix
    rate        : float [scalar > 0] => Time-Stretch -> rate > 1 ; Time-Compress -> rate < 1
    numPeaks    : int 2 < x < ? => Number of Peaks to Detect each frame
    hop_length  : int > 0 [scalar] or None
        The number of samples between successive columns of `D`.
        If None, defaults to `n_fft/4 = (D.shape[0]-1)/2`

        :param hop_length:
        :param numPeaks:
        :param rate:
        :param ySTFT:

    __Returns__
    yVocode     : np.ndarray [shape=(d, t / rate), dtype=complex] => phase vocoded STFT

    __Examples__
    # >>> # Time-Compress: Play at 1/2x speed
    # >>> rate = timeExpand = 0.5
    # >>> vxmod = 0.05     # Vox Modulation Depth - range[0.01, 1.0]
    # >>> ySTFT = stft(y)
    # >>> yVocodeOut = pvRobotSmith(ySTFT, rate, vxmod)
    # >>> yNewResult = istft(yVocodeOut, dtype=y.dtype)

    """

    n_fft = 2 * (ySTFT.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    time_steps = np.arange(0, ySTFT.shape[1], rate, dtype=float)

    # Create an empty output array
    yVocode = np.zeros((ySTFT.shape[0], len(time_steps)), ySTFT.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, ySTFT.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(ySTFT[:, 0])

    # Pad 0 columns to simplify boundary logic
    ySTFT = np.pad(ySTFT, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):
        columns = ySTFT[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = ((1.0 - alpha) * np.abs(columns[:, 0])
               + alpha * np.abs(columns[:, 1]))

        # Store to output array
        yVocode[:, t] = mag * np.exp(1.j * phase_acc * voxmod)

        # Compute phase advance
        dphase = (np.angle(columns[:, 1])
                  - np.angle(columns[:, 0])
                  - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return yVocode


# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


# reference C code

# int pva(float *input, float *window, float *output,
#        int input_size, int fftsize, int hopsize, float sr){
#
# int posin, posout, i, k, mod;
# float *sigframe, *specframe, *lastph;
# float fac, scal, phi, mag, delta, pi = (float)twopi/2;
#
# sigframe = new float[fftsize];
# specframe = new float[fftsize];
# lastph = new float[fftsize/2];
# memset(lastph, 0, sizeof(float)*fftsize/2);
#
# fac = (float) (sr/(hopsize*twopi));
# scal = (float) (twopi*hopsize/fftsize);
#
# for(posin=posout=0; posin < input_size; posin+=hopsize){
#      mod = posin%fftsize;
#	   # window & rotate a signal frame
#      for(i=0; i < fftsize; i++)
#          if(posin+i < input_size)
#            sigframe[(i+mod)%fftsize]
#                     = input[posin+i]*window[i];
#           else sigframe[(i+mod)%fftsize] = 0;
#
#      // transform it
#      fft(sigframe, specframe, fftsize);
#
#      // convert to PV output
#      for(i=2,k=1; i < fftsize; i+=2, k++){
#
#      // rectangular to polar
#      mag = (float) sqrt(specframe[i]*specframe[i] +
#                        specframe[i+1]*specframe[i+1]);
#      phi = (float) atan2(specframe[i+1], specframe[i]);
#      // phase diffs
#      delta = phi - lastph[k];
#      lastph[k] = phi;
#
#      // unwrap the difference, so it lies between -pi and pi
#      while(delta > pi) delta -= (float) twopi;
#      while(delta < -pi) delta += (float) twopi;
#
#      // construct the amplitude-frequency pairs
#      specframe[i] = mag;
#	   specframe[i+1] = (delta + k*scal)*fac;
#
#      }
#      // output it
#      for(i=0; i < fftsize; i++, posout++)
#			  output[posout] = specframe[i];
#
# }
# delete[] sigframe;
# delete[] specframe;
# delete[] lastph;
#
# return posout;
# }


# int pvs(float* input, float* window, float* output,
#          int input_size, int fftsize, int hopsize, float sr){
#
# int posin, posout, k, i, output_size, mod;
# float *sigframe, *specframe, *lastph;
# float fac, scal, phi, mag, delta;
#
# sigframe = new float[fftsize];
# specframe = new float[fftsize];
# lastph = new float[fftsize/2];
# memset(lastph, 0, sizeof(float)*fftsize/2);
#
# output_size = input_size*hopsize/fftsize;
#
# fac = (float) (hopsize*twopi/sr);
# scal = sr/fftsize;
#
# for(posout=posin=0; posout < output_size; posout+=hopsize){
#
#   // load in a spectral frame from input
#   for(i=0; i < fftsize; i++, posin++)
#        specframe[i] = input[posin];
#
# // convert from PV input to DFT coordinates
# for(i=2,k=1; i < fftsize; i+=2, k++){
#   delta = (specframe[i+1] - k*scal)*fac;
#   phi = lastph[k]+delta;
#   lastph[k] = phi;
#   mag = specframe[i];
#
#  specframe[i] = (float) (mag*cos(phi));
#  specframe[i+1] = (float) (mag*sin(phi));
#
# }
#   // inverse-transform it
#   ifft(specframe, sigframe, fftsize);
#
#   // unrotate and window it and overlap-add it
#   mod = posout%fftsize;
#   for(i=0; i < fftsize; i++)
#       if(posout+i < output_size)
#          output[posout+i] += sigframe[(i+mod)%fftsize]*window[i];
# }
# delete[] sigframe;
# delete[] specframe;
# delete[] lastph;
#
# return output_size;
# }