from matplotlib import pyplot as plt

class MaraCheckpointLoggedData(object):
    def __init__(self, fname):
        print "loading", fname
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
        self.kin     = array([entry["energies"]["kinetic" ] for entry in log])
        self.tie     = array([entry["energies"]["internal"] for entry in log])
        self.mag     = array([entry["energies"]["magnetic"] for entry in log])
        self.tot     = array([entry["energies"]["total"   ] for entry in log])
        self.mean_gamma = array([entry["mean_velocity"][0] for entry in log])
        self.max_gamma = array([entry["max_lorentz_factor"] for entry in log])
        self.runargs = runargs
        h5f.close()

    def plot_fields(self, fields, plot=None):
        plot = plot if plot else plt.plot
        for field in fields:
            y = getattr(self, field)
            plot(self.time, y, label=field)
        plt.xlabel(r"$t$")
        plt.ylabel(r"energy")
        plt.legend(loc='best')
        plt.show()
