import numpy as np
import h5py
import matplotlib.pyplot as plt
from toolbase import MaraTool
from autolog import logmethod

class MaraReductionsReader(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._h5file = h5py.File(filename, 'r')

    def available_power_spectra(self):
        for dset in self._h5file:
            if dset.startswith('pspec'):
                print dset, self._h5file[dset].keys()

    @logmethod
    def plot_power_spectra(self, which=['magnetic-solenoidal',
                                        'magnetic-dilatational',
                                        'velocity-solenoidal',
                                        'velocity-dilatational'],
                           tmin=0.0, tmax=-1.0):
        if not which: which = ['magnetic-solenoidal']
        for dset in self._h5file:
            if dset.startswith('pspec'):
                for w in which:
                    x = self._h5file[dset][w]['binloc']
                    y = self._h5file[dset][w]['binval']
                    try:
                        t = self._h5file[dset][w]['time'].value
                        label = r'%s $t=%f$'%(w, t)
                    except KeyError:
                        t = float(pspec[7:])
                        label = dset + " " + w
                    if tmax < 0.0 or (tmin < t < tmax):
                        plt.loglog(x, y, label=label)
        plt.legend(loc='best')
        plt.show()
