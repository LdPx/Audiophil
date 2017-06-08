import logging
from collections import namedtuple
import numpy as np

AudioFeatures = namedtuple('AudioFeatures', 'loudness zcr')

class AudioFeaturesCalculator(object):
    
    def calc(self, audio_data, framesize, framestep):
        return AudioFeatures(
            np.array([self.loudness(f) for f in self.frames(audio_data.data, framesize, framestep)]),
            np.array([self.zero_crossing_rate(f) for f in self.frames(audio_data.data, framesize, framestep)])),
    
    def frames(self, data, framesize, framestep):
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            yield data[i:i+framesize]
            
    def loudness(self, frame):
        return np.sum(frame ** 2) / len(frame)
    
    def zero_crossing_rate(self, frame):
        return np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))) / (2 * len(frame))
    
    
    