import h5py
from autolog import logmethod

class MaraCheckpointCreator(object):
    """
    Creates a new Mara checkpoint from scratch
    """
    @logmethod
    def __init__(self, filename, shape, mhd=True):
        import datetime
        chkpt = h5py.File(filename, 'w')
        status = chkpt.require_group("status")
        prim = chkpt.require_group("prim")
        measure = chkpt.require_dataset("measure", [], dtype='|S2')
        measure[...] = '[]' # json representation of an empty Lua table

        status['Checkpoint'] = 0.0
        status['CurrentTime'] = 0.0
        status['Iteration'] = 0.0
        status['LastMeasurementTime'] = 0.0
        status['Timestep'] = 0.0

        dsetnames = ['rho', 'pre', 'vx', 'vy', 'vz']
        if mhd: dsetnames += ['Bx', 'By', 'Bz']

        for dsetname in dsetnames:
            prim.require_dataset(dsetname, shape, dtype=float)

        chkpt['time_created'] = str(datetime.datetime.now())

        self._chkpt = chkpt

    @logmethod
    def apply_initial_data(self, callback):
        callback(self._chkpt['prim'])

    @logmethod
    def set_comment(self, comment):
        self._chkpt["comment"] = comment

    @logmethod
    def set_numpy_random_state(self, random_state):
        import pickle
        self._chkpt["numpy_random_state"] = pickle.dumps(random_state)
