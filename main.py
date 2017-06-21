import os
import pprint
import logging
import numpy as np
import matplotlib.pyplot as plt
from src.audio_features_calculator import AudioFeaturesCalculator
from src.audio_data_reader import AudioDataReader

# TODO fft aus audio_data_reader und audio_data rausnehmen - nah...

def plot_audio_features(audio_features, audio_data):
    for i, audio_feature in enumerate(audio_features):
        fig = plt.figure(i)
        # plt.title(audio_data[i].name)
        fig.canvas.set_window_title(audio_data[i].name)
        
        plt.subplot(321)
        plt.plot(audio_data[i].data)
        plt.ylabel("amplitude")
        plt.xlabel("t")
        
        plt.subplot(322)
        plt.plot(audio_data[i].fdata)
        plt.ylabel("energy")
        plt.xlabel("f")
        
        plt.subplot(323)
        plt.plot(audio_feature.loudness)
        plt.ylabel("loudness")
        plt.xlabel("frame")
        
        plt.subplot(324)                    
        plt.plot(audio_feature.zcr)
        plt.ylabel("zero_crossing_rate")                
        plt.xlabel("frame")                  
        
        plt.subplot(325)
        plt.plot(audio_feature.brightness)
        plt.ylabel("Spectral Centroid (/brightness)")
        plt.xlabel("frame")
        
        plt.subplot(326)
        plt.plot(audio_feature.ss)
        plt.ylabel("Spectral Spread (/Bandbreite)")
        plt.xlabel("frame")
        
        #plt.get_yaxis().get_major_formatter().set_useOffset(False) 
        #y_formatter = plt.ticker.ScalarFormatter(useOffset=False)
        #ax.yaxis.set_major_formatter(y_formatter)
        
    plt.show()

def main():        
    logging.basicConfig(level=logging.DEBUG)    
    data_folder = os.path.join(os.getcwd(), "data")    
    print('reading audio data')
    # WavFileWarning: Chunk (non-data) not understood, skipping it. WavFileWarning
    # there is some metadata in a file that is not understood by scipy.io.wavfile.read (->audio_data_reader.py)
    audio_data = AudioDataReader().read(data_folder)
    pprint.pprint(audio_data)
    
    framesize = 1024
    framestep = int(framesize / 2)
    print('calculating audio features')
    audio_features = [AudioFeaturesCalculator().calc(audio, framesize, framestep) for audio in audio_data]
    plot_audio_features(audio_features, audio_data)
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