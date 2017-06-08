import os
import pprint
import logging
import matplotlib.pyplot as plt
from src.audio_features_calculator import AudioFeaturesCalculator
from src.audio_data_reader import AudioDataReader

# TODO fft aus audio_data_reader und audio_data rausnehmen

def plot_audio_data(audio_data):
    for i, audio_datum in enumerate(audio_data):
        plt.figure(i)
        plt.title(audio_datum.name)
        plt.subplot(211)
        plt.plot(audio_datum.data)
        plt.ylabel("ampl")
        plt.xlabel("time")
        plt.subplot(212)
        plt.plot(abs(audio_datum.fdata))
        plt.ylabel("energy")
        plt.xlabel("freq")
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
    audio_data = AudioDataReader().read(data_folder)
    pprint.pprint(audio_data)    
    framesize = 1024
    framestep = int(framesize / 2)
    print('calculating audio features')
    audio_features = [AudioFeaturesCalculator().calc(ad, framesize, framestep) for ad in audio_data]
    pprint.pprint(audio_features)
    

if __name__ == '__main__':
    main()