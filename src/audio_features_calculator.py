import logging
from collections import namedtuple
import numpy as np

AudioFeatures = namedtuple('AudioFeatures', 'loudness zcr')

class AudioFeaturesCalculator(object):
    
    def calc(self, audio_data, framesize, framestep):
        # entspricht f_k
        frames = self.frames(audio_data.data, framesize, framestep)
        # entspricht X aller Frames
        # rfft: liefert 1. Hälfte des Frequenzspektrums
        frame_cmplx_comps = [np.fft.rfft(self.hamming_window(frame)) for frame in frames]
        # entspricht E aller Frames
        frame_energy_comps = [np.log10(np.abs(frame)) * 10 for frame in frame_cmplx_comps]
        frame_frequencies = [self.frame_freqs(frame, audio_data.samplerate) for frame in frames]
        return AudioFeatures(
            np.array([self.loudness(frame) for frame in frames]),
            np.array([self.zero_crossing_rate(frame) for frame in frames])),
    

    def frames(self, data, framesize, framestep):
        frames = []
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            frames.append(data[i:i+framesize])
            
        return frames
    
            
    def frame_freqs(self, frame, samplerate):
        indices = np.arange(0, len(frame))
        return indices * samplerate / len(frame)
    
            
    def loudness(self, frame):
        return np.sum(frame ** 2) / len(frame)
    
    
    def zero_crossing_rate(self, frame):
        return np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))) / (2 * len(frame))
    
    
    def hamming_window(self, frame):
        indices = np.arange(0, len(frame))
        return frame * (0.54 - 0.46 * np.cos(2*np.pi*indices/(len(frame)-1))) 
    
    