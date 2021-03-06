from autolog import logmethod

class MaraCheckpointLoggedData(object):

    @logmethod
    def __init__(self, fname):
        from h5py import File
        from numpy import array, log10
        import json

        h5f = File(fname, "r")
        log = sorted(json.loads(h5f["measure"].value).values(),
                 key=lambda e: e["Status"]["Iteration"])
        runargs = json.loads(h5f["runargs"].value)
        self.N       = runargs["N"]
        self.zeta    = runargs.get("zeta", 1.0)
        self.time    = array([entry["Status" ]["CurrentTime"] for entry in log])
        self.mean_T  = array([entry["mean_T" ] for entry in log])
        self.max_T   = array([entry["max_T"  ] for entry in log])
        self.mean_Ms = array([entry["mean_Ms"] for entry in log])
        self.max_Ms  = array([entry["max_Ms" ] for entry in log])
        self.mean_Ma = array([entry["mean_Ma"] for entry in log])
        self.min_Ma  = array([entry["min_Ma" ] for entry in log])
        self.kin     = array([entry["energies"]["kinetic" ] for entry in log])
        self.tie     = array([entry["energies"]["internal"] for entry in log])
        self.mag     = array([entry["energies"]["magnetic"] for entry in log])
        self.tot     = array([entry["energies"]["total"   ] for entry in log])
        self.mean_gamma = array([entry["mean_velocity"][0] for entry in log])
        self.max_gamma = array([entry["max_lorentz_factor"] for entry in log])
        self.runargs = runargs
        h5f.close()

    def plot_fields(self, fields, plot_axis=None, noshow=False, **plot_args):
        from matplotlib import pyplot as plt

        if plot_axis is None:
            fig = plt.figure()
            ax1 = fig.add_subplot('111')
        else:
            ax1 = plot_axis

        for field in fields:
            y = getattr(self, field)
            plt.loglog(self.time, y, label=field, **plot_args)
            #plt.plot(self.time, y, label=field, **plot_args)
        #ax1.loglog(self.time[1:], 1e-3*self.time[1:]**(-4./3.),
        #           ls='--', c='k', label=r'$t^{-4/3}$')
        #ax1.loglog(self.time[1:], 1e-2*self.time[1:]**(-3./3.),
        #           ls='--', c='k', label=r'$t^{-1}$')
        ax1.set_xlabel(r"$t$")
        ax1.set_ylabel(r"energy")

        if plot_axis is None:
            ax1.legend(loc='best')
        if not noshow:
            plt.show()
