import os
import glob
import errno
import random
import urllib.request as request
import numpy as np
from scipy.io import loadmat


class CWRU:

    def __init__(self, exp, rpm, length, root=os.path.join(os.path.expanduser('~'), 'Datasets/CWRU')):
        if exp not in ('12DriveEndFault', '12FanEndFault', '48DriveEndFault'):
            print ("wrong experiment name: {}".format(exp))
            exit(1)
        if rpm not in ('1797', '1772', '1750', '1730'):
            print( "wrong rpm value: {}".format(rpm))
            exit(1)
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
        self._load_and_slice_data(rdir, lines)
        # shuffle training and test arrays
        self._shuffle()
        self.labels = tuple(line[2] for line in lines)
        self.nclasses = len(self.labels)  # number of classes

    def _mkdir(self, path):
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

    def _load_and_slice_data(self, rdir, infos):
        self.x = np.zeros((0, self.length))
        self.y = np.zeros(0, dtype=np.longlong)
        for idx, info in enumerate(infos):
            # directory of this file
            fdir = os.path.join(rdir, info[0], info[1])
            self._mkdir(fdir)
            fpath = os.path.join(fdir, info[2] + '.mat')
            if not os.path.exists(fpath):
                self._download(fpath, info[3].rstrip('\n'))

            mat_dict = loadmat(fpath)
            key = list(filter(lambda x: 'DE_time' in x, mat_dict.keys()))[0]
            time_series = mat_dict[key][:, 0]

            idx_last = -(time_series.shape[0] % self.length)
            clips = time_series[:idx_last].reshape(-1, self.length)

            n = clips.shape[0]
            self.x = np.vstack((self.x, clips))
            self.y = np.hstack((self.y, np.full(n, idx, dtype=np.longlong)))

    def _shuffle(self):
        # shuffle training samples
        index = list(range(self.x.shape[0]))
        random.Random(0).shuffle(index)
        self.x = self.x[index]
        self.y = self.y[index]
