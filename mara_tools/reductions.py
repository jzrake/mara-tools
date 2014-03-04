import numpy as np
import h5py
from toolbase import MaraTool
from autolog import logmethod

class MaraReductionsReader(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._h5file = h5py.File(filename, 'r')

    def available_power_spectra(self):
        for dset in self._h5file:
            if not dset.startswith('pspec'): continue
            print dset, self._h5file[dset].keys()

    @logmethod
    def spectrum_near_time(self, time, which):
        time_dict = { }
        for dset in self._h5file:
            if not dset.startswith('pspec'): continue
            time_dict[self._h5file[dset][which]['time'].value] = dset
        times = np.array(time_dict.keys())
        tsnap = times[np.argmin(abs(times - time))]
        print "snapped from t=%3.2f to t=%3.2f" % (time, tsnap)
        return time_dict[tsnap]

    @logmethod
    def plot_power_spectra(self, which=['magnetic-solenoidal',
                                        'magnetic-dilatational',
                                        'velocity-solenoidal',
                                        'velocity-dilatational'],
                           tmin=0.0, tmax=-1.0,
                           nolegend=False,
                           title='',
                           hardcopy='',
                           skip=1):
        import matplotlib.pyplot as plt
        if not which: which = ['magnetic-solenoidal']
        for n, dset in enumerate(self._h5file):
            if not dset.startswith('pspec'): continue
            if not n % skip == 0: continue
            for w in which:
                x = self._h5file[dset][w]['binloc']
                y = self._h5file[dset][w]['binval']
                t = self._h5file[dset][w]['time'].value
                label = r'%s $t=%f$'%(w, t)
                if tmax < 0.0 or (tmin < t < tmax):
                    plt.loglog(x, y, label=label)
        if not nolegend:
            plt.legend(loc='best')
        plt.xlabel(r'$k$')
        plt.ylabel(r'$dP/dk$')
        plt.title(title)
        if hardcopy:
            plt.savefig(hardcopy)
        else:
            plt.show()

    @logmethod
    def plot_single_power_spectrum(self, time, which, comp=0.0, **plot_args):
        import matplotlib.pyplot as plt
        dset = self.spectrum_near_time(time, which)
        N = len(self._h5file[dset][which]['binloc'])
        x = self._h5file[dset][which]['binloc'][:N/2]
        y = self._h5file[dset][which]['binval'][:N/2]
        t = self._h5file[dset][which]['time'].value
        if 'label' not in plot_args:
            plot_args['label'] = r'%s:%s $t=%f$'%(self._h5file.filename,
                                                  which, t)
        plt.loglog(x, y*x**comp, **plot_args)
        return x, y

    @logmethod
    def plot_time_devel(self, which=['magnetic-solenoidal',
                                     'magnetic-dilatational',
                                     'velocity-solenoidal',
                                     'velocity-dilatational'],
                        kbins=[8,32,64],
                        tmin=0.0, tmax=-1.0,
                        nolegend=False,
                        title='',
                        hardcopy=''):
        import matplotlib.pyplot as plt
        if not which: which = ['magnetic-solenoidal']
        if not kbins: kbins = [32]

        for kbin in kbins:
            for w in which:
                ts = [ ]
                Ps = [ ]
                for dset in self._h5file:
                    if not dset.startswith('pspec'): continue
                    ts.append(self._h5file[dset][w]['time'].value)
                    Ps.append(self._h5file[dset][w]['binval'][kbin])
                    k0 = self._h5file[dset][w]['binloc'][kbin]
                plt.loglog(ts, Ps, label=r"%s: $k=%3.2f$" % (w, k0))
        plt.xlabel(r'$t$')
        plt.ylabel(r'$P_k(t)$')
        plt.legend(loc='best')
        plt.show()

