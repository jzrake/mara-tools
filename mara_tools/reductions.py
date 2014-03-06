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
                           tmin=0.0, tmax=1e10,
                           nolegend=False,
                           title='',
                           hardcopy='',
                           skip=1,
                           cmap=''):
        import matplotlib as mpl
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt

        if not which: which = ['magnetic-solenoidal']

        xs = [ ]
        ys = [ ]
        ts = [ ]
        ws = [ ]

        for n, dset in enumerate(self._h5file):
            if not dset.startswith('pspec'): continue
            if not n % skip == 0: continue
            for w in which:
                x = self._h5file[dset][w]['binloc']
                y = self._h5file[dset][w]['binval']
                t = self._h5file[dset][w]['time'].value
                if tmin <= t <= tmax:
                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                    ws.append(w)

        if cmap:
            norm = mpl.colors.Normalize(vmin=ts[0], vmax=ts[-1])
            cmap = cm.ScalarMappable(norm=norm, cmap=getattr(cm, cmap))

        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        for x, y, t, w in zip(xs, ys, ts, ws):
            label = r'%s $t=%f$'%(w, t)
            kwargs = dict(label=label,
                          alpha=self.get_user_param('alpha', 1.0))
            if cmap: kwargs['c'] = cmap.to_rgba(t)
            ax1.loglog(x, y, **kwargs)

        if not nolegend:
            plt.legend(loc='best')

        if self.get_user_param('xlim', False):
            ax1.set_xlim(self.get_user_param('xlim', False))
        if self.get_user_param('ylim', False):
            ax1.set_ylim(self.get_user_param('ylim', False))
        ax1.set_xlabel(r'$k$')
        ax1.set_ylabel(r'$dP/dk$')
        ax1.set_title(title)
        if hardcopy:
            plt.savefig(hardcopy)
        else:
            plt.show()

    @logmethod
    def plot_single_power_spectrum(self, time, which, comp=0.0, plot_axis=None,
                                   **plot_args):
        import matplotlib.pyplot as plt
        dset = self.spectrum_near_time(time, which)
        N = len(self._h5file[dset][which]['binloc'])
        x = self._h5file[dset][which]['binloc'][:N/2]
        y = self._h5file[dset][which]['binval'][:N/2]
        t = self._h5file[dset][which]['time'].value
        if 'label' not in plot_args:
            plot_args['label'] = r'%s:%s $t=%f$'%(self._h5file.filename,
                                                  which, t)
        if plot_axis is None:
            fig = plt.figure()
            ax1 = fig.add_subplot('111')
        else:
            ax1 = plot_axis
        ax1.loglog(x, y*x**comp, **plot_args)
        ax1.set_xlabel(r'$k L/2\pi$')
        ax1.set_ylabel(r'$dP/dk$')
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

    @logmethod
    def dump_npz(self):
        dsets = { }
        def cb(name, obj):
            if isinstance(obj, h5py.Dataset):
                if obj.shape:
                    val = obj.value
                else:
                    val = [obj.value]
                dsets[name.replace('/', '.')] = val
        self._h5file.visititems(cb)
        np.savez(self._h5file.filename.replace('.h5', '.npz'), **dsets)
