# -*- coding: utf-8 -*-
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header begin-----------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************
#
# __::((xodmaVocoder_tb.py))::__
#
# ___::((XODMK Programming Industries))::___
# ___::((XODMK:CGBW:BarutanBreaks:djoto:2020))::___
#
#
# XODMK Phase Vocoder testbench
#
#
#
# *****************************************************************************
# /////////////////////////////////////////////////////////////////////////////
# header end-------------------------------------------------------------------
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# *****************************************************************************

import os
import sys
import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt


rootDir = '../../'

audioSrcDir = rootDir+'audio/wavsrc/'
audioOutDir = rootDir+'audio/test/'



sys.path.insert(0, rootDir+'audio/xodma')

from xodmaAudioTools import load_wav, write_wav, valid_audio, resample
from xodmaAudioTools import samples_to_time, time_to_samples, fix_length
from xodmaSpectralTools import amplitude_to_db, stft, istft, peak_pick
from xodmaSpectralTools import magphase
from xodmaVocoder import pvTimeStretch, pvPitchShift
from xodmaSpectralUtil import frames_to_time
from xodmaSpectralPlot import specshow

#sys.path.insert(0, 'C:/odmkDev/odmkCode/odmkPython/util')
sys.path.insert(1, rootDir+'util')
import xodPlotUtil as xodplt



#sys.path.insert(1, 'C:/odmkDev/odmkCode/odmkPython/DSP')
sys.path.insert(2, rootDir+'DSP')
import xodClocks as clks


# temp python debugger - use >>>pdb.set_trace() to set break
import pdb

# // *---------------------------------------------------------------------* //

plt.close('all')

# /////////////////////////////////////////////////////////////////////////////
# #############################################################################
# begin : function definitions
# #############################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


# // *---------------------------------------------------------------------* //


def arrayFromFile(fname):
    ''' reads .dat data into Numpy array:
        fname is the name of existing file in dataInDir (defined above)
        example: newArray = arrayFromFile('mydata_in.dat') '''
        
    fileSrcFull = audioSrcDir+fname
        
    datalist = []
    with open(fileSrcFull, mode='r') as infile:
        for line in infile.readlines():
            datalist.append(float(line))
    arrayNm = np.array(datalist)
    
    fileSrc = os.path.split(fileSrcFull)[1]
    # src_path = os.path.split(sinesrc)[0]
    
    print('\nLoaded file: '+fileSrc)
    
    lgth1 = len(list(arrayNm))    # get length by iterating csvin obj (only way?)
    print('Length of data = '+str(lgth1))
    
    return arrayNm



# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


print('// //////////////////////////////////////////////////////////////// //')
print('// *--------------------------------------------------------------* //')
print('// *---::ODMK Spectral Tools Experiments::---*')
print('// *--------------------------------------------------------------* //')
print('// \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ //')


# // *---------------------------------------------------------------------* //
# // *--User Settings - Primary parameters--*
# // *---------------------------------------------------------------------* //

# srcSel: 0 = wavSrc, 1 = amenBreak, 2 = sineWave48K, 
#         3 = multiSin test, 4 = text array input

srcSel =  0


# STEREO source signal
#wavSrc = 'dsvco.wav'
#wavSrc = 'detectiveOctoSpace_one.wav'
#wavSrc = 'ebolaCallibriscian_uCCrhythm.wav'
#wavSrc = 'zoroastorian_mdychaos1.wav'
#wavSrc = 'The_Amen_Break_odmk.wav'

# MONO source signal
#wavSrc = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'

wavSrcA = 'opium_house_1.wav'
#wavSrcB = 'scoolreaktor_beatx03.wav'
#wavSrcB = 'gorgulans_beatx01.wav'

# length of input signal:
# '0'   => full length of input .wav file
# '###' => usr defined length in SECONDS
wavLength = 0



NFFT = 2048
STFTHOP = int(NFFT/4)
WIN = 'hann'


''' Valid Window Types: 

boxcar
triang
blackman
hamming
hann
bartlett
flattop
parzen
bohman
blackmanharris
nuttall
barthann
kaiser (needs beta)
gaussian (needs standard deviation)
general_gaussian (needs power, width)
slepian (needs width)
dpss (needs normalized half-bandwidth)
chebwin (needs attenuation)
exponential (needs decay scale)
tukey (needs taper fraction)

'''

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


# inputs:  wavIn, audioSrcDir, wavLength
# outputs: ySrc_ch1, ySrc_ch2, numChannels, fs, ySamples


# Load Stereo/mono .wav file


if (srcSel==0):
    srcANm = wavSrcA
elif (srcSel==1):
    srcANm = 'The_Amen_Break_48K.wav'
elif (srcSel==2):
    srcANm = 'MonoSinOut_48K_560Hz_5p6sec.wav'
elif (srcSel==3):
    srcANm = 'multiSinOut48KHz_1K_3K_5K_7K_9K_16sec.wav'
    
audioSrcA = audioSrcDir+srcANm

#audioSrcB = audioSrcDir+wavSrcB


    
[aSrc, aNumChannels, afs, aLength, aSamples] = load_wav(audioSrcA, wavLength, True)

#[bSrc, bNumChannels, bfs, bLength, bSamples] = load_wav(audioSrcB, wavLength, True)



if aNumChannels == 2:
    aSrc_ch1 = aSrc[:,0];
    aSrc_ch2 = aSrc[:,1];
else:
    aSrc_ch1 = aSrc;
    aSrc_ch2 = 0;

#if bNumChannels == 2:
#    bSrc_ch1 = bSrc[:,0];
#    bSrc_ch2 = bSrc[:,1];
#else:
#    bSrc_ch1 = bSrc;
#    bSrc_ch2 = 0;



#aT = 1.0 / afs
#print('\nsample period: ------------------------- '+str(aT))
#print('wav file datatype: '+str(sf.info(audioSrcA).subtype))


# // *--- Plot - source signal ---*

if 1:
    
    fnum = 3
    pltTitle = 'Input Signals: aSrc_ch1'
    pltXlabel = 'sinArray time-domain wav'
    pltYlabel = 'Magnitude'
    
    # define a linear space from 0 to 1/2 Fs for x-axis:
    xaxis = np.linspace(0, len(aSrc_ch1), len(aSrc_ch1))
    
    xodplt.xodPlot1D(fnum, aSrc_ch1, xaxis, pltTitle, pltXlabel, pltYlabel)
        
    plt.show()

#pdb.set_trace()

# // *---------------------------------------------------------------------* //
# // *---------------------------------------------------------------------* //


if 1:

    print('\n')
    print('// *---:: Phase Vocoder Time-Stretch EFX test ::---*')
    
    '''
    Compress to be twice as fast

    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> y_fast = librosa.effects.time_stretch(y, 2.0)

    Or half the original speed

    >>> y_slow = librosa.effects.time_stretch(y, 0.5) '''
    

    # Time Compress rate:
    R = 1.23
    
    # Time Expand rate:
    S = 0.3
    

    yRxFast_ch1 = pvTimeStretch(aSrc_ch1, 2.0)
    yRxFast_ch2 = pvTimeStretch(aSrc_ch2, 2.0)
    
    yRxFast = np.transpose( np.column_stack((yRxFast_ch1, yRxFast_ch2)) )
    
    print('\nPerformed time_stretch by R (Rxfast)')
    
    #pdb.set_trace()
    
    ySxSlow_ch1 = pvTimeStretch(aSrc_ch1, 0.5)
    ySxSlow_ch2 = pvTimeStretch(aSrc_ch2, 0.5)   

    ySxSlow = np.transpose( np.column_stack((ySxSlow_ch1, ySxSlow_ch2)) )
    
    print('\nPerformed time_stretch by S (Sxslow)')


    print('\n// *---:: Write .wav files ::---*')

    outFilePath = audioOutDir+'yOriginal.wav'
    write_wav(outFilePath, aSrc, afs)


    outFilePath = audioOutDir+'yRxFast.wav'
    write_wav(outFilePath, yRxFast, afs)
    
    
    outFilePath = audioOutDir+'ySxSlow.wav'
    write_wav(outFilePath, ySxSlow, afs)

    print('\n\nOutput directory: '+audioOutDir)
    print('\nwrote .wav file yOriginal.wav')
    print('\nwrote .wav file yRxFast.wav')
    print('\nwrote .wav file ySxSlow.wav')

  
    
# // *---------------------------------------------------------------------* //

plt.show()

print('\n')
print('// *--------------------------------------------------------------* //')
print('// *---::done::---*')
print('// *--------------------------------------------------------------* //')

# // *---------------------------------------------------------------------* //




# reference C code


#int pva(float *input, float *window, float *output, 
#        int input_size, int fftsize, int hopsize, float sr){
#
#int posin, posout, i, k, mod;
#float *sigframe, *specframe, *lastph;
#float fac, scal, phi, mag, delta, pi = (float)twopi/2;
#
#sigframe = new float[fftsize];
#specframe = new float[fftsize];
#lastph = new float[fftsize/2];
#memset(lastph, 0, sizeof(float)*fftsize/2);
#
#fac = (float) (sr/(hopsize*twopi));
#scal = (float) (twopi*hopsize/fftsize);
#
#for(posin=posout=0; posin < input_size; posin+=hopsize){
#      mod = posin%fftsize;
#	// window & rotate a signal frame
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
#	  specframe[i+1] = (delta + k*scal)*fac;
#
#      }
#      // output it
#      for(i=0; i < fftsize; i++, posout++)
#			  output[posout] = specframe[i];
#		  
#}
#delete[] sigframe;
#delete[] specframe;
#delete[] lastph;
#
#return posout;
#}


#int pvs(float* input, float* window, float* output,
#          int input_size, int fftsize, int hopsize, float sr){
#
#int posin, posout, k, i, output_size, mod;
#float *sigframe, *specframe, *lastph;
#float fac, scal, phi, mag, delta;
#
#sigframe = new float[fftsize];
#specframe = new float[fftsize];
#lastph = new float[fftsize/2];
#memset(lastph, 0, sizeof(float)*fftsize/2);
#
#output_size = input_size*hopsize/fftsize;
#
#fac = (float) (hopsize*twopi/sr);
#scal = sr/fftsize;
#
#for(posout=posin=0; posout < output_size; posout+=hopsize){ 
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
#}
#   // inverse-transform it
#   ifft(specframe, sigframe, fftsize);
#
#   // unrotate and window it and overlap-add it
#   mod = posout%fftsize;
#   for(i=0; i < fftsize; i++)
#       if(posout+i < output_size)
#          output[posout+i] += sigframe[(i+mod)%fftsize]*window[i];
#}
#delete[] sigframe;
#delete[] specframe;
#delete[] lastph;
#
#return output_size;
#}