import logging
from collections import namedtuple
import numpy as np

AudioFeatures = namedtuple('AudioFeatures', 'loudness zcr')

class AudioFeaturesCalculator(object):
    
    def calc(self, audio_data, framesize, framestep):
        # rfft: liefert 1. Hälfte des Frequenzspektrums
        frame_cmplx_comps = [np.fft.rfft(self.hamming_window(f)) for f in self.frames(audio_data.data, framesize, framestep)]
        #logging.debug('calced complex frame components {}'.format(frame_cmplx_comps))
        frame_energy_comps = [np.log10(np.abs(f)) * 10 for f in frame_cmplx_comps]
        #logging.debug('calced energy frame components {}'.format(frame_energy_comps))
        return AudioFeatures(
            np.array([self.loudness(f) for f in self.frames(audio_data.data, framesize, framestep)]),
            np.array([self.zero_crossing_rate(f) for f in self.frames(audio_data.data, framesize, framestep)])),
    
    # TODO Liste
    def frames(self, data, framesize, framestep):
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            yield data[i:i+framesize]
            
    def loudness(self, frame):
        return np.sum(frame ** 2) / len(frame)
    
    def zero_crossing_rate(self, frame):
        return np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))) / (2 * len(frame))
    
    def hamming_window(self, frame):
        indices = np.arange(0, len(frame))
        return frame * (0.54 - 0.46 * np.cos(2*np.pi*indices/(len(frame)-1))) 
    
    