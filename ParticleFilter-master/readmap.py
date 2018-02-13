import re
from scipy.stats import norm, uniform, expon
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Map:
    def __init__(self, filename):
        self.meta = self.read_meta_data(filename)
        self.map = self.read_map(filename)

    def read_meta_data(self, filename):
        meta = {}
        f = open(filename, 'r')
        head = [next(f) for x in xrange(6)]

        for line in head:
            line = re.sub('robot_specifications->', '', line)
            tmp = re.sub('\s+', ' ', line).split()
            if len(tmp) == 2:
                meta[tmp[0]] = int(tmp[1])

        line = next(f)
        tmp = re.sub('\s+', ' ', line).split()
        meta['rows'] = int(tmp[1])
        meta['cols'] = int(tmp[2])
        f.close()
        return meta

    def read_map(self, filename):

        data = []
        tmp_map = np.zeros(shape=(self.meta['rows'],self.meta['cols']))
        with open(filename, 'r') as f:
            data = f.readlines();
        data = data[7:]  # skip header
        for i, row in enumerate(data):
            values = row.split()
            for j, val in enumerate(values):
                tmp_map[i][j] = float(val)

        return tmp_map

    def visualize(self):
        plt.imsave('filename.png', self.map, cmap=cm.gray)
