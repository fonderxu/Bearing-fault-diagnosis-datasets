
import os
import errno
import random
import urllib.request as request
import numpy as np
from scipy.io import loadmat


class Cwru:
    """CWRU Dataset.

      dataset link:  https://pan.xunlei.com/s/VNNVs4cGhuL1NqGrauTU19zYA1?pwd=kta7# extract code: kta7
      Args:
          root (str): Root directory of dataset
          exp (str): experiment state of dataset
          rpm (str): The task (domain) to create dataset. Choices include ``'600'``, \
              ``'800'`` and ``'1000'``.
          length (int): window length (or num of vibrations of each input)
          hop_size (int): hop length (or num of vibrations of each input
          shuffle (bool): whether shuffle dataset
      """

    sample_location = [
        "12DriveEndFault",
        "12FanEndFault",
        "48DriveEndFault"
    ]

    domain_list = [
        "1797",
        "1772",
        "1750",
        "1730",
    ]

    CLASSES = {
        '12DriveEndFault_1797': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', '0.028-Ball', '0.028-InnerRace', 'Normal'),
        '12DriveEndFault_1772': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', '0.028-Ball', '0.028-InnerRace', 'Normal'),
        '12DriveEndFault_1750': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', '0.028-Ball', '0.028-InnerRace', 'Normal'),
        '12DriveEndFault_1730': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', '0.028-Ball', '0.028-InnerRace', 'Normal'),
        '12FanEndFault_1797': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace3', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace',
            '0.021-OuterRace6',
            'Normal'),
        '12FanEndFault_1772': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace3', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace3', 'Normal'),
        '12FanEndFault_1750': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace3', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace3', 'Normal'),
        '12FanEndFault_1730': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace3', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace3', 'Normal'),
        '48DriveEndFault_1797': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', 'Normal'),
        '48DriveEndFault_1772': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', 'Normal'),
        '48DriveEndFault_1750': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', 'Normal'),
        '48DriveEndFault_1730': (
            '0.007-Ball', '0.007-InnerRace', '0.007-OuterRace12', '0.007-OuterRace3', '0.007-OuterRace6', '0.014-Ball',
            '0.014-InnerRace', '0.014-OuterRace6', '0.021-Ball', '0.021-InnerRace', '0.021-OuterRace12',
            '0.021-OuterRace3',
            '0.021-OuterRace6', 'Normal'),
    }

    def __init__(self, root: str, exp: str, rpm: str, length=1024, hop_size=1024, shuffle=False) -> None:
        assert exp in self.sample_location and rpm in self.domain_list
        super(Cwru, self).__init__()
        # root directory of all data
        rdir = os.path.join(root, exp, rpm)
        fmeta = os.path.join(os.path.dirname(__file__), 'metadata.txt')
        all_lines = open(fmeta).readlines()
        lines = []
        for line in all_lines:
            l = line.split()
            if (l[0] == exp or l[0] == 'NormalBaseline') and l[1] == rpm:
                lines.append(l)

        self.length = length  # sequence length
        self.hop_size = hop_size
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        if shuffle:
            self._shuffle()
        self.labels = self.CLASSES[f'{exp}_{rpm}']
        self.nclasses = len(self.labels)  # number of classes

    def _mkdir(self, path) -> None:
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                print("can't create directory '{}''".format(path))
                exit(1)

    def _download(self, fpath, link):
        print("Downloading to: '{}'".format(fpath))
        request.urlretrieve(link, fpath)

    def _load_and_slice_data(self, rdir, infos) -> None:
        self.x = np.zeros((0, self.length))
        self.y = np.zeros(0, dtype=np.longlong)
        for idx, info in enumerate(infos):
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))

            x, y = self._get_file_data(fpath, idx, self.length, self.hop_size)
            self.x = np.vstack((self.x, x))
            self.y = np.hstack((self.y, y))

    def _get_file_data(self, file, label, length, hop_size) -> tuple[np.ndarray, np.ndarray]:
        mat_dict = loadmat(file)
        key = list(filter(lambda x: 'DE_time' in x, mat_dict.keys()))[0]
        data = mat_dict[key][:, 0]

        samples = []
        start_index = 0
        while(start_index + length < data.size):
            samples.append(data[start_index: start_index + length])
            start_index += hop_size
        return np.vstack(samples).astype(np.float32), np.full(len(samples), label, dtype=np.longlong)

    def _shuffle(self) -> None:
        # shuffle training samples
        index = list(range(self.x.shape[0]))
        random.Random(0).shuffle(index)
        self.x = self.x[index]
        self.y = self.y[index]



