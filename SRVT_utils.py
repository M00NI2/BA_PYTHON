"""
@author: Kirill Shchegelskiy
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import os
import sys
import soundfile as sf
import pyloudnorm as pyln

def fileLoudness(path):
    """
    Determines loudness rating of a single audio file according to ITU-R BS.1770-4

    Returns value in dB LUFS
    """
    data, rate = sf.read(path)
    loudness = 0
    if np.shape(data)[0]>10000:
        #print('File Name {} data size {}'.format(Path(path).name[-19:-15], np.shape(data)[0]))
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
    return loudness

def fileFreqAnalysis(path):
    """
    Helper function for some spectral characteristics
    """
    x, fs = librosa.load(path, sr=None)
    S = np.abs(librosa.stft(x, n_fft=1024))
    fftFreq = librosa.fft_frequencies(sr=fs, n_fft=1024)

    Sp = np.sum(S, axis=1)
    indValidSp = np.flatnonzero(Sp > 10)
    if np.shape(indValidSp)[0] == 0:
        topFreq = 0
    else:
        topFreq = fftFreq[indValidSp[-1]]
    return topFreq, np.max(Sp), np.min(Sp)

def fileFreqBandBandwise(path):
    """
    Estimates bandwidth used in speech file recording by
    comparing average power spectrum bandwise to the 
    base band value and overflow value

    Returns assigned label or EMPTY for empty recording
    """
    x, fs = librosa.load(path, sr=48000)
    S = np.abs(librosa.stft(x, n_fft=1024)) 
    SdB = librosa.power_to_db(S**2)
    fftFreq = librosa.fft_frequencies(sr=fs, n_fft=1024)
    
    averageSdB = np.average(SdB, axis=1)
    baseBandSdB = averageSdB[(fftFreq > 800) & (fftFreq < 2500)] # 36 frequencies in range of this 'base' band
    WBandSdB = averageSdB[(fftFreq > 5000) & (fftFreq < 7000)] # 43 frequencies in range of this 'wide' band
    SWBandSdB = averageSdB[(fftFreq > 9000) & (fftFreq < 13000)] # 85 frequencies in range of this 'superwide' band
    FBandSdB = averageSdB[(fftFreq > 15000) & (fftFreq < 20000)] # 106 frequencies in range of this 'full' band
    overFlowSdB = averageSdB[fftFreq > 21000] # 64 frequencies in overflow range

    avBaseBandSdB = np.average(baseBandSdB)
    avWBandSdB = np.average(WBandSdB)
    avSWBandSdB = np.average(SWBandSdB)
    avFBandSdB = np.average(FBandSdB)
    avOverFlowSdB = np.average(overFlowSdB)
    #print('WB-OF {:.3f} SWB-OF {:.3f} FB-OF {:.3f}'.format(avWBandSdB-avOverFlowSdB, avSWBandSdB-avOverFlowSdB, avFBandSdB-avOverFlowSdB))
    
    if avBaseBandSdB-avOverFlowSdB < 1:
        return "EMPTY"
    elif avWBandSdB-avOverFlowSdB < 1:
        return "NB"
    elif avSWBandSdB-avOverFlowSdB < 1:
        return "WB"
    elif avFBandSdB-avOverFlowSdB < 1:
        return "SWB"
    else:
        return "FB"

def fileFreqBandConst(path, con=10):
    """
    Determines bandwidth used in speech file recording by 
    comparing signal energy frequency-wise to parameter con

    Returns assigned label or EMPTY for empty recording
    """
    x, fs = librosa.load(path, sr=None)
    S = np.abs(librosa.stft(x, n_fft=1024))
    fftFreq = librosa.fft_frequencies(sr=fs, n_fft=1024)

    Sp = np.sum(S, axis=1)
    indValidSp = np.flatnonzero(Sp > con)
    if np.shape(indValidSp)[0] == 0:
        return "EMPTY"
    elif fftFreq[indValidSp[-1]] < 4000:
        return "NB"
    elif fftFreq[indValidSp[-1]] < 8000:
        return "WB"
    elif fftFreq[indValidSp[-1]] < 16000:
        return "SWB"
    else:
        return "FB"

def fileFreqBandRelative(path, factor=1000):
    """
    Determines bandwidth used in speech file recording by
    comparing signal energy frequency-wise to a fraction
    1/factor of the maximum spectral peak

    Returns assigned label or EMPTY for empty recording
    """
    x, fs = librosa.load(path, sr=None)
    S = np.abs(librosa.stft(x, n_fft=1024))
    fftFreq = librosa.fft_frequencies(sr=fs, n_fft=1024)

    Sp = np.sum(S, axis=1)
    indValidSp = np.flatnonzero(Sp > np.max(Sp)/factor)
    if np.shape(indValidSp)[0] == 0:
        return "EMPTY"
    elif fftFreq[indValidSp[-1]] < 4000:
        return "NB"
    elif fftFreq[indValidSp[-1]] < 8000:
        return "WB"
    elif fftFreq[indValidSp[-1]] < 16000:
        return "SWB"
    else:
        return "FB"



def fileSARCounter(path):
    """
    Counts number of Short Amplitude Reversals in audio file

    SAR are signal abnormalities of 'wraparound' or 'overflow' type:
    signal amplitude instantly changes sign while being of high abs value

    Returns number of such occurences
    """
    x, fs = librosa.load(path, sr=None)
    burst_counter = 0
    xmax = 0
    second_reversal = False # flag for proper SAR
    reversal_end = 0 # to keep track of SAR edges

    if np.shape(x)[0] != 0:
        xmax = x.max()

    for n in range(np.shape(x)[0]-6):
        # check for sudden jump in signal AND end of one reversal can not be beginning of another
        if np.abs(x[n+1]-x[n]) > 1.5*xmax and reversal_end != n+1:
            for k in range(5): # checking with window size 5 for signal flipping back to normal
                if np.abs(x[n+k+2]-x[n+k+1]) > 1.5*xmax:
                    second_reversal = True
                    reversal_end = n+k+2
                    break 
        # if checked for proper SAR then counter goes up and flag goes down   
        if second_reversal:  
            burst_counter = burst_counter+1
            second_reversal = False

    return burst_counter

def fileSARPlotter(path):
    """
    Plots the audio signal and Short Amplitude Reversals

    SAR are signal abnormalities of 'wraparound' or 'overflow' type:
    signal amplitude instantly changes sign while being of high abs value
    """
    x, fs = librosa.load(path, sr=None)
    burst_counter = 0
    xmax = 0

    if np.shape(x)[0] != 0:
        xmax = x.max()

    bursts = np.empty(0, dtype=int)
    for n in range(np.shape(x)[0]-1):
        if np.abs(x[n+1]-x[n]) > 1.5*x.max: 
            if bursts.shape[0] == 0 or n-bursts[-1] > 5:
                bursts = np.append(bursts, n, axis=None)
            burst_counter = burst_counter+1
            
    burst_times = librosa.samples_to_time(bursts, sr=fs)
    #print(burst_counter, bursts)

    fig, ax = plt.subplots(nrows=2, sharex=True)
    librosa.display.waveshow(x, sr=fs, ax=ax[0], color='blue')
    ax[1].vlines(burst_times, 0, 1, color='r', alpha=0.8, label='SARs')
    ax[1].legend(frameon=True, framealpha=0.8)
    ax[1].label_outer()
    plt.show()

def fileSARRemoval(path):
    """
    Removes the Short Amplitude Reversals from audio file

    SAR are signal abnormalities of 'wraparound' or 'overflow' type:
    signal amplitude instantly changes sign while being of high abs value

    Creates a copy of original file speech.wav file named speech_SARrestored.wav
    """
    x, fs = librosa.load(path, sr=None)
    burst_counter = 0
    xmax = 0
    second_reversal = False
    reversal_end = 0

    if np.shape(x)[0] != 0:
        xmax = x.max()

    for n in range(np.shape(x)[0]-1):
        # check for sudden jump in signal AND end of one reversal can not be beginning of another
        if np.abs(x[n+1]-x[n]) > 1.5*xmax and reversal_end != n+1: 
            for k in range(5): # checking with window size 5 for signal flipping back to normal
                if np.abs(x[n+k+2]-x[n+k+1]) > 1.5*xmax:
                    second_reversal = True
                    reversal_end = n+k+2
                    break    
        if second_reversal:  
            x[n+1]=-x[n+1] # if we are sure that it is SAR, we flip signal back to 'original'
            burst_counter = burst_counter+1
            second_reversal = False
    
    if burst_counter > 0: # create new file only if we restored data
        sf.write(path[:-4]+'_SARrestored.wav', data=x, samplerate=fs, subtype='PCM_24')
        print('File Name {} restored'.format(Path(path).name[:4]))

def fileMeanTopContrast(path):
    """
    Helper function, computes spectral contrast of audio file
    
    Returns mean contrast value for highest frequency band

    Can be useful for high-freq noise detection
    """
    x, fs = librosa.load(path, sr=None)
    S = np.abs(librosa.stft(x))
    contrast = librosa.feature.spectral_contrast(S=S, sr=fs, fmin=312.5, n_bands=6)
    return np.mean(contrast[6,:])

def fileMFCC0(path):
    """
    Helper function, computes Mel-frequency cepstral coefficients of audio file

    Returns standard deviation of first MFCC
    """
    x, fs = librosa.load(path, sr=None)
    mfccs = librosa.feature.mfcc(y=x, sr=fs)
    return np.std(mfccs[0,:])

def wavFilter(filePath):
    if filePath.endswith(".wav"):
        return True
    else:
        return False
    
def main(args):
    dirPath = args[0]
    wavFilePaths = filter(wavFilter, os.listdir(dirPath))
    #for wavPath in wavFilePaths:
    #    loudness = fileLoudness(dirPath+"/"+wavPath)
    #    print('File Name {} Loudness {:.3f}'.format(Path(wavPath).name[4:8], loudness))

    #    topFreq, maxEnergy, minEnergy = fileFreqAnalysis(dirPath+"/"+wavPath)
    #    print('File Name {} Top Freq {} Min Energy {} Max Energy {}'.format(Path(wavPath).name[:4], topFreq, minEnergy, maxEnergy))

    #    bandEstimationBandwise = fileFreqBandBandwise(dirPath+"/"+wavPath)
    #    print('File Name {} Bandwidth Bandwise {}'.format(Path(wavPath).name[:4], bandEstimationBandwise))
    #    bandEstimationRelative = fileFreqBandRelative(dirPath+"/"+wavPath)
    #    bandEstimationConst = fileFreqBandConst(dirPath+"/"+wavPath)
    #    if bandEstimationConst != bandEstimationRelative:
    #        print('File Name {} BandwidthConst {} BandwidthRelative {}'.format(Path(wavPath).name[:4], bandEstimationConst, bandEstimationRelative))

    #    fileSARRemoval(dirPath+"/"+wavPath)
    #    burstCounter = fileSARCounter(dirPath+"/"+wavPath)
    #    print('File Name {} SAR number {}'.format(Path(wavPath).name[:4], burstCounter))

    #    mfcc0 = fileMFCC0(dirPath+"/"+wavPath)
    #    meanContrast = fileMeanTopContrast(dirPath+"/"+wavPath)
    #    print('File name {} MFCC[0] {:.6f}'.format(Path(wavPath).name[:4], mfcc0))
    #    print('File name {} {:.6f}'.format(Path(wavPath).name[:4], meanContrast))

if __name__ == '__main__':
    main(sys.argv[1:])

