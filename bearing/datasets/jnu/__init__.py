"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""

import os
import random
import numpy as np


class Jnu:
    """Jnu Dataset.

    dataset link: https://pan.xunlei.com/s/VNNVrpOoaEAPPdIODQaJQQcQA1?pwd=jnri# extract code: jnri

    Args:
        root (str): Root directory of dataset
        rpm (str): The task (domain) to create dataset. Choices include ``'600'``, \
            ``'800'`` and ``'1000'``.
        length (int): window length (or num of vibrations of each input)
        hop_size (int): hop length (or num of vibrations of each input
        shuffle (bool): whether shuffle dataset
    """
    domain_list = {
        "600": ['n600.csv', 'ib600.csv', 'ob600.csv', 'tb600.csv'],
        "800": ['n800.csv', 'ib800.csv', 'ob800.csv', 'tb800.csv'],
        "1000": ['n1000.csv', 'ib1000.csv', 'ob1000.csv', 'tb1000.csv']
    }
    CLASSES = ['Normal Condition', 'Inner Ball', 'Outer Fault', 'Ball Fault']

    def __init__(self, root: str, rpm: str, length=1024, hop_size=1024, shuffle=False) -> None:
        assert rpm in self.domain_list.keys()
        super(Jnu, self).__init__()
        self.x, self.y = self._get_domain_data(root, rpm, length, hop_size)

        if shuffle:
            self._shuffle()
        self.classes = self.CLASSES
        self.num_classes = len(self.classes)

    def _get_domain_data(self, root, rpm, length, hop_size) -> tuple[np.ndarray, np.ndarray]:
        files = [os.path.join(root, file) for file in self.domain_list[rpm]]
        samples, targets = [], []

        for index, file in enumerate(files):
            sample, target = self._get_file_data(file, index, length, hop_size)
            samples.append(sample), targets.append(target)
        return np.vstack(samples), np.concatenate(targets, axis=0)

    def _get_file_data(self, file, label, length, hop_size) -> tuple[np.ndarray, np.ndarray]:
        samples = []
        data = np.loadtxt(file)
        start_index = 0
        while(start_index + length < data.size):
            samples.append(data[start_index: start_index + length])
            start_index += hop_size
        return np.vstack(samples).astype(np.float32), np.full(len(samples), label, dtype=np.longlong)

    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.x.shape[0]))
        random.Random(0).shuffle(index)
        self.x = self.x[index]
        self.y = self.y[index]
