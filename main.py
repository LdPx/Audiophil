import os
import pprint
import logging
from src.audio_data_reader import AudioDataReader
import matplotlib.pyplot as plt
#from numpy.fft import irfft

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
    logging.basicConfig(level=logging.DEBUG)    
    data_folder = os.path.join(os.getcwd(), "data")
    audio_data = AudioDataReader().read(data_folder)    
    print('read audio data')
    pprint.pprint(audio_data)    
    framesize = 1024
    framestep = int(framesize / 2)
    #pprint.pprint(list(AudioDataReader().get_frames(audio_data[0].data, framesize, framestep)))
    #plot_audio_data(audio_data)
    

if __name__ == '__main__':
    main()