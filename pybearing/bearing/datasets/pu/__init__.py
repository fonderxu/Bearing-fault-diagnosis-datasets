__all__ = ['Pu']

import os
import random
import numpy as np
from scipy.io import loadmat


class Pu_Utils(object):
    bearing_condition = {
        'K001': {'Damage_Component': 'NO', 'Damage_Level': 'NO', 'Damage_Method': 'NO'},
        'K002': {'Damage_Component': 'NO', 'Damage_Level': 'NO', 'Damage_Method': 'NO'},
        'K003': {'Damage_Component': 'NO', 'Damage_Level': 'NO', 'Damage_Method': 'NO'},
        'K004': {'Damage_Component': 'NO', 'Damage_Level': 'NO', 'Damage_Method': 'NO'},
        'K005': {'Damage_Component': 'NO', 'Damage_Level': 'NO', 'Damage_Method': 'NO'},
        'K006': {'Damage_Component': 'NO', 'Damage_Level': 'NO', 'Damage_Method': 'NO'},
        'KA01': {'Damage_Component': 'OR', 'Damage_Level': '1', 'Damage_Method': 'EDM'},
        'KA03': {'Damage_Component': 'OR', 'Damage_Level': '2', 'Damage_Method': 'electric engraver'},
        'KA05': {'Damage_Component': 'OR', 'Damage_Level': '1', 'Damage_Method': 'electric engraver'},
        'KA06': {'Damage_Component': 'OR', 'Damage_Level': '2', 'Damage_Method': 'electric engraver'},
        'KA07': {'Damage_Component': 'OR', 'Damage_Level': '1', 'Damage_Method': 'drilling'},
        'KA08': {'Damage_Component': 'OR', 'Damage_Level': '2', 'Damage_Method': 'drilling'},
        'KA09': {'Damage_Component': 'OR', 'Damage_Level': '2', 'Damage_Method': 'drilling'},
        'KI01': {'Damage_Component': 'IR', 'Damage_Level': '1', 'Damage_Method': 'EDM'},
        'KI03': {'Damage_Component': 'IR', 'Damage_Level': '1', 'Damage_Method': 'electric engraver'},
        'KI05': {'Damage_Component': 'IR', 'Damage_Level': '1', 'Damage_Method': 'electric engraver'},
        'KI07': {'Damage_Component': 'IR', 'Damage_Level': '2', 'Damage_Method': 'electric engraver'},
        'KI08': {'Damage_Component': 'IR', 'Damage_Level': '2', 'Damage_Method': 'electric engraver'},
    }

    def __init__(self):
        super(Pu_Utils, self).__init__()

    @staticmethod
    def _get_file_labels(root, work_state):
        def get_label(value):
            Label = {
                'NO_NO': 0,
                'OR_1': 1,
                'OR_2': 2,
                'IR_1': 3,
                'IR_2': 4,
            }
            key = value['Damage_Component'] + '_' + value['Damage_Level']
            return Label[key]

        file_labels = []
        for key in Pu_Utils.bearing_condition:
            path, label = os.path.join(root, key, f'{work_state}_{key}_1.mat'), get_label(
                Pu_Utils.bearing_condition[key])
            file_labels.append((path, label))
        return file_labels

    @staticmethod
    def _get_data(file, label, length, hop_size):
        fl = loadmat(file)[file.split(os.sep)[-1].split('.')[0]]
        fl = fl[0][0][2][0][6][2]  # Take out the data
        fl = fl.reshape(-1)
        samples = []
        start_index = 0
        while (start_index + length < fl.size):
            samples.append(fl[start_index: start_index + length])
            start_index += hop_size
        return np.vstack(samples).astype(np.float32), np.full(len(samples), label, dtype=np.longlong)

    @staticmethod
    def get_sample_labels(root, work_state, length, hop_size):
        file_labels = Pu_Utils._get_file_labels(root, work_state)
        samples, labels = [], []
        for file, label in file_labels:
            sample, label = Pu_Utils._get_data(file, label, length, hop_size)
            samples.append(sample)
            labels.append(label)
        return np.concatenate(samples, axis=0), np.hstack(labels)


class Pu:
    """Pu Dataset.
      dataset link: https://www.aliyundrive.com/s/AanRmmBNZna  extract code: 4qq9
      Args:
          root (str): Root directory of dataset
          work_state (str): experiment work_state of dataset
          length (int): window length (or num of vibrations of each input)
          hop_size (int): hop length (or num of vibrations of each input
          shuffle (bool): whether shuffle dataset
      """

    work_state = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
    CLASSES = [
        'Normal',
        'OR-1',
        'OR-2',
        'IR-1',
        'IR-2',
    ]

    def __init__(self, root: str, work_state: str, length=1024, hop_size=1024, shuffle=False):
        assert work_state in self.work_state
        super(Pu, self).__init__()

        self.x, self.y = Pu_Utils.get_sample_labels(root, work_state, length, hop_size)
        if shuffle:
            self._shuffle()
        self.classes = Pu.CLASSES
        self.num_classes = len(self.classes)

    def _shuffle(self) -> None:
        # shuffle training samples
        index = list(range(self.x.shape[0]))
        random.Random(0).shuffle(index)
        self.x = self.x[index]
        self.y = self.y[index]