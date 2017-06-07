import os
import pprint
from src.audio_data_reader import AudioDataReader
import matplotlib.pyplot as plt


def main():
    data_folder = os.path.join(os.getcwd(), "data")
    audio_data = AudioDataReader().read(data_folder)    
    print('read audio data:\n{}'.format(pprint.pformat(audio_data)))
    for i, audio_datum in enumerate(audio_data):
        plt.figure(i)
        plt.plot(audio_datum.data)
        plt.ylabel("amplitude")
        plt.xlabel("time")
        plt.title(audio_datum.name)
    plt.show()

if __name__ == '__main__':
    main()