import os
import logging
from collections import namedtuple
from scipy.io import wavfile
import numpy as np

AudioData = namedtuple('AudioData', 'name data fdata samplerate')


class AudioDataReader(object):
    
    def read(self, data_folder):
        audio_data = []
        for root, _dirs, files in os.walk(data_folder):
            for file in sorted(files):
                res = wavfile.read(os.path.join(root, file))
                logging.info('read file {}'.format(file))
                audio_data.append(AudioData(name=file, data=res[1], fdata=np.fft.rfft(res[1]), samplerate=res[0]))
                
        return audio_data
    
    
    def get_frames(self, data, framesize, framestep):
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            yield data[i:i+framesize]
    