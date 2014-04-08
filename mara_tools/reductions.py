import numpy as np
import h5py
from toolbase import MaraTool
from autolog import logmethod


class MaraReductionsReader(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._h5file = h5py.File(filename, 'r')
        self._time_dict_cached = False

    @logmethod
    def cache_time_dict(self):
        which = 'magnetic-solenoidal'
        self._time_dict = { }
        for dset in self._h5file:
            self._time_dict[self._h5file[dset][which]['time'].value] = dset
        self._times = np.array(self._time_dict.keys())
        self._time_dict_cached = True

    def available_power_spectra(self):
        for t, ds in sorted(self._time_dict.items()):
            yield t, ds

    @logmethod
    def spectrum_near_time(self, time):
        if not self._time_dict_cached:
            self.cache_time_dict()
        times = self._times
        tsnap = times[np.argmin(abs(times - time))]
        print "snapped from t=%3.2f to t=%3.2f" % (time, tsnap)
        return self._time_dict[tsnap]

    def peak_power_at_time(self, time, which):
        dset = self.spectrum_near_time(time)
        N = len(self._h5file[dset][which]['binloc'])
        t = self._h5file[dset][which]['time'].value
        k = self._h5file[dset][which]['binloc'][:N/2]
        P = self._h5file[dset][which]['binval'][:N/2]
        n = np.argmax(y)
        return k[n], P[n]

    @logmethod
    def peak_power_all(self, which, skip=1):
        ts, ks, Ps = [], [], []
        for n, dset in enumerate(self._h5file):
            if n % skip != 0: continue
            N = len(self._h5file[dset][which]['binloc'])
            t = self._h5file[dset][which]['time'].value
            k = self._h5file[dset][which]['binloc'][:N/2]
            P = self._h5file[dset][which]['binval'][:N/2]
            n = np.argmax(P)
            ts.append(t)
            ks.append(k[n])
            Ps.append(P[n])
        data = sorted(zip(ts, ks, Ps))
        return ([t for t, k, P in data],
                [k for t, k, P in data],
                [P for t, k, P in data])

    @logmethod
    def rms_wavenumber_all(self, which, skip=1):
        """
        WARNING! Assumes equally-spaced wavenumber bins
        """
        ts, ks, Ps = [], [], []
        for n, dset in enumerate(self._h5file):
            if n % skip != 0: continue
            N = len(self._h5file[dset][which]['binloc'])
            t = self._h5file[dset][which]['time'].value
            k = self._h5file[dset][which]['binloc'][:N/2]
            P = self._h5file[dset][which]['binval'][:N/2]
            ts.append(t)
            ks.append((k*P).sum() / P.sum())
            Ps.append(0.0)
        data = sorted(zip(ts, ks, Ps))
        return (np.array([t for t, k, P in data]),
                np.array([k for t, k, P in data]),
                np.array([P for t, k, P in data]))

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

        comp = self.get_user_param('comp', 0.0)
        kcut = self.get_user_param('kcut', None)

        for n, dset in enumerate(self._h5file):
            if not dset.startswith('pspec'): continue
            if not n % skip == 0: continue
            for w in which:
                x = self._h5file[dset][w]['binloc'].value
                y = self._h5file[dset][w]['binval'].value
                t = self._h5file[dset][w]['time'].value

                if kcut:
                    I = (kcut[0] < x) *  (x < kcut[1])
                    x = x[I]
                    y = y[I]

                if tmin <= t <= tmax:
                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                    ws.append(w)

        if cmap:
            norm = mpl.colors.Normalize(vmin=ts[0], vmax=ts[-1])
            cmap = cm.ScalarMappable(norm=norm, cmap=getattr(cm, cmap))

        ax1 = self.get_plot_axes()

        for x, y, t, w in zip(xs, ys, ts, ws):
            label = r'%s $t=%f$'%(w, t)
            kwargs = dict(label=label,
                          alpha=self.get_user_param('alpha', 1.0))
            if cmap: kwargs['c'] = cmap.to_rgba(t)
            ax1.loglog(x, y*x**comp, **kwargs)

        if not nolegend:
            plt.legend(loc='best')

        if self.get_user_param('xlim', False):
            ax1.set_xlim(self.get_user_param('xlim', False))
        if self.get_user_param('ylim', False):
            ax1.set_ylim(self.get_user_param('ylim', False))
        ylabel = self.get_user_param('ylabel', r'$dP/dk$')

        ax1.set_xlabel(r'$kL/2\pi$', fontsize=18)
        ax1.set_ylabel(ylabel, fontsize=18)
        ax1.set_title(title)

        self.show()

    @logmethod
    def plot_single_power_spectrum(self, time, which, comp=0.0,
                                   **plot_args):
        import matplotlib.pyplot as plt
        dset = self.spectrum_near_time(time)
        N = len(self._h5file[dset][which]['binloc'])
        x = self._h5file[dset][which]['binloc'][:N/2]
        y = self._h5file[dset][which]['binval'][:N/2]
        t = self._h5file[dset][which]['time'].value
        if 'label' not in plot_args:
            plot_args['label'] = r'%s:%s $t=%f$'%(self._h5file.filename,
                                                  which, t)
        ax1 = self.get_plot_axes()
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
