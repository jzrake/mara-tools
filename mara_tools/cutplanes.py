import numpy as np
import h5py
import matplotlib.pyplot as plt
from toolbase import MaraTool
from autolog import logmethod

class MaraCheckpointCutplaneExtractor(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._chkpt = h5py.File(filename, 'r')

    @logmethod
    def plot_slice(self, field='rho', cmap='jet', axis=0, index=0):
        if axis == 0: imgdata = self._chkpt['prim'][field][index,:,:]
        if axis == 1: imgdata = self._chkpt['prim'][field][:,index,:]
        if axis == 2: imgdata = self._chkpt['prim'][field][:,:,index]

        fig = plt.figure(figsize=[10,10])
        sax = fig.add_subplot('111')
        cax = sax.imshow(imgdata, origin='image', interpolation='nearest')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        fig.colorbar(cax, shrink=0.85, pad=0.0, aspect=20, cmap=cmap,
                     orientation="horizontal")
        plt.setp(sax.get_xticklabels(), visible=False)
        plt.setp(sax.get_yticklabels(), visible=False)
        plt.show()
