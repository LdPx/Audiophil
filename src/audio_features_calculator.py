import logging
from collections import namedtuple
import numpy as np

AudioFeatures = namedtuple('AudioFeatures', 'loudness')

class AudioFeaturesCalculator(object):
    
    def calc(self, audio_data, framesize, framestep):
        return AudioFeatures(self.loudness(audio_data, framesize, framestep))
    
    def frames(self, data, framesize, framestep):
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            yield data[i:i+framesize]
            
    def loudness(self, audio_data, framesize, framestep):
        return np.array([np.sum(frame ** 2) for frame in self.frames(audio_data.data, framesize, framestep)])
    