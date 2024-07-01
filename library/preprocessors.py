from pyeeglab import Preprocessor
import json
import logging
import numpy as np
import pandas as pd
from math import floor
from typing import List
import warnings
from mne.io import Raw


warnings.filterwarnings('ignore')

class ZScoreNormalization(Preprocessor):

    def run(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        mean_v = kwargs['mean']
        std_v = kwargs['std']
        return (data - mean_v)/std_v

class BandPassFrequency_new(Preprocessor):

    def __init__(self, low_freq: float, high_freq: float, low_stop: float, high_stop: float, phase: 'str') -> None:
        super().__init__()
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.low_stop = low_stop
        self.high_stop = high_stop
        self.phase = phase
        logging.debug('Create band pass filter (fs1 %f Hz, fp1 %f Hz, fp2 %f Hz, fs2 %f Hz),  preprocessor', low_stop, low_freq, high_freq, high_stop)

    def to_json(self) -> str:
        out = {
            self.__class__.__name__ : {
                'low_freq': self.low_freq,
                'high_freq': self.high_freq,
                'low_stop': self.low_stop,
                'high_stop': self.high_stop,
                'phase': self.phase
            }
        }
        out = json.dumps(out)
        return out

    def run(self, data: Raw, **kwargs) -> Raw:
        return data.filter(self.low_freq, self.high_freq, l_trans_bandwidth = self.low_stop, h_trans_bandwidth = self.high_stop, phase = self.phase)


class NotchFrequency_new(Preprocessor):

    def __init__(self, freq: float, phase: 'str') -> None:
        super().__init__()
        self.freq = freq
        self.phase = phase
        logging.debug('Create notch filter %f Hz preprocessor', freq)

    def to_json(self) -> str:
        out = {
            self.__class__.__name__ : {
                'freq': self.freq,
                'phase': self.phase
            }
        }
        out = json.dumps(out)
        return out

    def run(self, data: Raw, **kwargs) -> Raw:
        notch_f = np.arange(self.freq, (self.freq*2)+1, self.freq)
        return data.notch_filter(notch_f, phase = self.phase)


class ClassDivision(Preprocessor):

    def run(self, data:Raw, **kwargs) -> pd.DataFrame:

        ch_set = kwargs['channels_set']
        interval = kwargs['interval']
        curr_sj = kwargs['ID'] ##newadd
        logging.debug(curr_sj) ##newadd
        dur = data.n_times/data.info['sfreq'] ##

        list_crop = []
        list_elem = []
        for elem in interval:
            temp = data.copy()
            list_elem.append(elem[1])
            
            tmax_adjusted = min(elem[1], dur - 0.01)  # Subtract a small buffer to ensure no overflow

            # Crop the data with the adjusted tmax
            crop = temp.crop(tmin=elem[0], tmax=tmax_adjusted).get_data(units='uV')
            #crop = temp.crop(elem[0], elem[1]).get_data(units= 'uV')

            list_crop.append(pd.DataFrame(crop))
        final = pd.concat([dff for dff in list_crop], axis = 1)
        final = final.T
        final = final.set_axis(ch_set, axis = 1)
        return final

class from_Raw_to_np(Preprocessor):
    def run(self, data:Raw, **kwargs) -> np.array:
        final= data.get_data()
        return final

class StaticWindow_new(Preprocessor):

    def __init__(self, length: float) -> None:
        super().__init__()
        self.length = length
        logging.debug('Create static frames of (%d seconds each) generator', length)

    def to_json(self) -> str:
        out = {
            self.__class__.__name__ : {
                'length': self.length
            }
        }
        out = json.dumps(out)
        return out

    def run(self, data: pd.DataFrame, **kwargs) -> List[pd.DataFrame]:
        step = floor(self.length * kwargs['lowest_frequency'])

        return [data[t:t+step] for t in range(0, len(data) - step, step)]
    