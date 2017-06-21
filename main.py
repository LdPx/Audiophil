import os
import pprint
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.audio_features_calculator import AudioFeaturesCalculator
from src.audio_data_reader import AudioDataReader

# TODO fft aus audio_data_reader und audio_data rausnehmen - nah...

def plot_audio_data(audio_data):
    for i, audio_datum in enumerate(audio_data):
        plt.figure(i)
        plt.title(audio_datum.name)
        
        plt.subplot(221)
        plt.plot(audio_datum.data)
        plt.ylabel("ampl")
        plt.xlabel("time")
        
        plt.subplot(222)                    # 
        plt.plot(abs(audio_datum.fdata))    # Scratch
        plt.ylabel("energy")                #  that
        plt.xlabel("freq")                  # vvvvvv
        
        #=======================================================================
        # plt.subplot(222)
        # plt.plot(abs(audio_datum.fdata))
        # plt.ylabel("Zero-Crossing-Rate")
        # plt.xlabel("time")
        #=======================================================================
        
        plt.subplot(223)
        plt.plot(1)
        plt.ylabel("Spectral Centroid")
        plt.xlabel("time")
        
        plt.subplot(224)
        plt.plot(1)
        plt.ylabel("Spectral Spread")
        plt.xlabel("time")
        
    #===========================================================================
    # for i, audio_datum in enumerate(audio_data):
    #     res = audio_datum.fdata
    #     res[abs(res) < 0.25*10e6] = 0
    #     res = irfft(res)
    #     plt.figure(i + len(audio_data) * 2)
    #     plt.plot(res)
    #     plt.ylabel("ampl")
    #     plt.xlabel("time")
    #     plt.title(audio_datum.name)
    #===========================================================================
    plt.show()

def main():        
    logging.basicConfig(level=logging.INFO)    
    data_folder = os.path.join(os.getcwd(), "data")    
    print('reading audio data')
    # WavFileWarning: Chunk (non-data) not understood, skipping it. WavFileWarning
    # there is some metadata in a file that is not understood by scipy.io.wavfile.read (->audio_data_reader.py)
    audio_data = AudioDataReader().read(data_folder)
    pprint.pprint(audio_data)
    plot_audio_data(audio_data)
    
    framesize = 1024
    framestep = int(framesize / 2)
    print('calculating audio features')
    audio_features = [AudioFeaturesCalculator().calc(audio, framesize, framestep) for audio in audio_data]
    #pprint.pprint(audio_features)
    
    for i,audio_feature in enumerate(audio_features):
        numzero, numinf, numnan = 0,0,0
        for feature in audio_feature:
            numzero += (feature == 0).sum()
            numinf += np.isinf(feature).sum()
            numnan += np.isnan(feature).sum()
        print('datum {}: {} zeros, {} +/-infs, {} nans'.format(i, numzero, numinf, numnan))

if __name__ == '__main__':
    main()