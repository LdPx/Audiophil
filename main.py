import os
import pprint
from src.audio_data import AudioData
from scipy.io import wavfile


def main():
    data_folder = os.path.join(os.getcwd(), "data")
    audio_data = []
    for root, _dirs, files in os.walk(data_folder):
        for file in sorted(files):
            res = wavfile.read(os.path.join(root, file))
            audio_data.append(AudioData(name=file, data=res[1], samplerate=res[0]))
    
    print('read audio data:\n{}'.format(pprint.pformat(audio_data)))

if __name__ == '__main__':
    main()