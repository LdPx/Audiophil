import logging
from collections import namedtuple
import numpy as np

AudioFeatures = namedtuple('AudioFeatures', 'loudness zcr')

class AudioFeaturesCalculator(object):
    
    def calc(self, audio_data, framesize, framestep):
        return AudioFeatures(
            self.loudness(audio_data, framesize, framestep),
            self.zero_crossing_rate(audio_data, framesize, framestep))
    
    def frames(self, data, framesize, framestep):
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            yield data[i:i+framesize]
            
    def loudness(self, audio_data, framesize, framestep):
        return np.array([np.sum(frame ** 2) / len(frame) for frame in self.frames(audio_data.data, framesize, framestep)])
    
    def zero_crossing_rate(self, audio_data, framesize, framestep):
        frame_zero_crossing_rates = []
        for frame in self.frames(audio_data.data, framesize, framestep):
            frame_zero_crossing_rates.append(np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))) / (2 * len(frame)))
        return np.array(frame_zero_crossing_rates)
    
    
    