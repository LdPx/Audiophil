import os
from collections import namedtuple
from scipy.io import wavfile

AudioData = namedtuple('AudioData', 'name data samplerate')


class AudioDataReader(object):
    
    def read(self, data_folder):
        audio_data = []
        for root, _dirs, files in os.walk(data_folder):
            for file in sorted(files):
                res = wavfile.read(os.path.join(root, file))
                audio_data.append(AudioData(name=file, data=res[1], samplerate=res[0]))
                
        return audio_data