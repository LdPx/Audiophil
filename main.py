import os
import pprint
from src.audio_data_reader import AudioDataReader
import matplotlib.pyplot as plt
from numpy.fft import irfft


def main():
    data_folder = os.path.join(os.getcwd(), "data")
    audio_data = AudioDataReader().read(data_folder)    
    print('read audio data:\n{}'.format(pprint.pformat(audio_data)))
    for i, audio_datum in enumerate(audio_data):
        plt.figure(i)
        plt.plot(audio_datum.data)
        plt.ylabel("ampl")
        plt.xlabel("time")
        plt.title(audio_datum.name)
    for i, audio_datum in enumerate(audio_data):
        plt.figure(i + len(audio_data))
        plt.plot(abs(audio_datum.fdata))
        plt.ylabel("energy")
        plt.xlabel("freq")
        plt.title(audio_datum.name)
    
    for i, audio_datum in enumerate(audio_data):
        res = audio_datum.fdata
        res[abs(res) < 0.25*10e6] = 0
        res = irfft(res)
        plt.figure(i + len(audio_data) * 2)
        plt.plot(res)
        plt.ylabel("ampl")
        plt.xlabel("time")
        plt.title(audio_datum.name)
    plt.show()

if __name__ == '__main__':
    main()