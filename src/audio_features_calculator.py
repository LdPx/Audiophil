import logging
from collections import namedtuple
import numpy as np

AudioFeatures = namedtuple('AudioFeatures', 'loudness zcr brightness ss')

class AudioFeaturesCalculator(object):
    
    def calc(self, audio_data, framesize, framestep):        
        frames = self.frames(audio_data.data, framesize, framestep)
        # entspricht X aller Frames; rfft liefert 1. Hälfte des Frequenzspektrums
        frame_cmplx_comps = [np.fft.rfft(self.hamming_window(frame)) for frame in frames]
        # entspricht E aller Frames
        frame_energy_comps = [10 * np.log10(np.abs(frame)) for frame in frame_cmplx_comps]
        # entspricht f_k
        frame_frequencies = [self.frame_freqs(len(frame), audio_data.samplerate) for frame in frame_cmplx_comps]

        loudness = np.array([self.loudness(frame) for frame in frames])
        zero_crossing_rate = np.array([self.zero_crossing_rate(frame) for frame in frames])
        brightness = np.array([self.brightness(ff, fe) for ff,fe in zip(frame_frequencies,frame_energy_comps)])
        spectral_spread = np.array([self.spectral_spread(ff, fe, b) for ff,fe,b in zip(frame_frequencies,frame_energy_comps,brightness)])
        return AudioFeatures(loudness, zero_crossing_rate, brightness, spectral_spread)


    def frames(self, data, framesize, framestep):
        frames = []
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            frames.append(data[i:i+framesize])
            
        return frames
    
    
    def hamming_window(self, frame):
        indices = np.arange(0, len(frame))
        return frame * (0.54 - 0.46 * np.cos(2*np.pi*indices/(len(frame)-1))) 
    
            
    def frame_freqs(self, framesize, samplerate):
        indices = np.arange(0, framesize)
        return indices * samplerate / framesize
    
            
    def loudness(self, frame):
        return np.sum(frame ** 2) / len(frame)
    
    
    def zero_crossing_rate(self, frame):
        return np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))) / (2 * len(frame))
    
    
    def brightness(self, frame_frequencies, frame_energies):
        if -np.inf in frame_energies:
            return -np.inf
        return np.sum(frame_frequencies * frame_energies) / np.sum(frame_energies)
    
    
    def spectral_spread(self, frame_frequencies, frame_energies, brightness):
        if -np.inf in frame_energies:
            return -np.inf
        return np.sum((frame_frequencies - brightness) ** 2 * frame_energies) / np.sum(frame_energies)
    
    
    
    
    
    
    
    