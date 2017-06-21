import logging
from collections import namedtuple
import numpy as np

AudioFeatures = namedtuple('AudioFeatures', 'loudness zcr brightness ss')

class AudioFeaturesCalculator(object):
    
    def calc(self, audio_data, framesize, framestep):        
        loudness, zero_crossing_rate, brightness, spectral_spread = [],[],[],[]
        for frame in self.frames(audio_data.data, framesize, framestep):
            
            loudness.append(self.loudness(frame))
            zero_crossing_rate.append(self.zero_crossing_rate(frame))
            
            # entspricht X aller Frames; rfft liefert 1. Hälfte des Frequenzspektrums
            frame_cmplx_comps = np.fft.rfft(self.hamming_window(frame))
            # entspricht E aller Frames
            frame_energy_comps = np.abs(frame_cmplx_comps)
            #frame_energy_comps[frame_energy_comps == 0] = 1 # ersetze, sodass log(...) = 0 statt -inf
            frame_energy_comps = 10 * np.log10(frame_energy_comps)
            #frame_energy_comps = 10 * np.log10(np.abs(frame_cmplx_comps))
            # entspricht f_k aller Frames
            frame_frequencies = self.frame_freqs(len(frame_cmplx_comps), audio_data.samplerate)
    
            brightness.append(self.brightness(frame_frequencies, frame_energy_comps))
            spectral_spread.append(self.spectral_spread(frame_frequencies, frame_energy_comps, brightness[-1]))
            logging.debug('features of frame: {}'.format((loudness[-1],zero_crossing_rate[-1],brightness[-1],spectral_spread[-1])))
            
        return AudioFeatures(np.array(loudness), np.array(zero_crossing_rate), np.array(brightness), np.array(spectral_spread))


    def frames(self, data, framesize, framestep):
        for i in range(0, len(data)-(framesize-1), framestep):
            logging.debug('yielding frame {}'.format(range(i,i+framesize)))
            yield data[i:i+framesize]
    
    
    def hamming_window(self, frame):
        indices = np.arange(0, len(frame))
        return frame * (0.54 - 0.46 * np.cos(2*np.pi*indices/(len(frame)-1)))
    
            
    def frame_freqs(self, framesize, samplerate):
        indices = np.arange(0, framesize)
        return indices * samplerate / framesize
    
            
    def loudness(self, frame):
        #=======================================================================
        # sum = 0
        # for dat in frame:
        #     sum += (dat * dat)
        # 
        # return (sum / len(frame))
        #=======================================================python-tk================
        return np.sum(frame ** 2) / len(frame)
    
    
    def zero_crossing_rate(self, frame):
        return np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))) / (2 * len(frame))
    
    
    def brightness(self, frame_frequencies, frame_energies):
        fe_sum = np.sum(frame_energies)
        #if fe_sum == 0:
        #    return 0
        return np.sum(frame_frequencies * frame_energies) / fe_sum
    
    
    def spectral_spread(self, frame_frequencies, frame_energies, brightness):
        #if brightness == 0:
        #    return 0
        # logging.debug('ff {}'.format(frame_frequencies))
        # logging.debug('br {}'.format(brightness))
        return np.sum((frame_frequencies - brightness) ** 2 * frame_energies) / np.sum(frame_energies)
    
    
    
    
    
    
    