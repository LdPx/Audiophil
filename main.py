import os
import pprint
from src.audio_data_reader import AudioDataReader


def main():
    data_folder = os.path.join(os.getcwd(), "data")
    audio_data = AudioDataReader().read(data_folder)
    
    print('read audio data:\n{}'.format(pprint.pformat(audio_data)))

if __name__ == '__main__':
    main()