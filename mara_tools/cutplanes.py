import inspect
import numpy as np
import h5py
import matplotlib.pyplot as plt
from autolog import logmethod

class MaraCheckpointCutplaneExtractor(object):
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

    @classmethod
    def populate_parser(self, method, parser):
        f = getattr(self, method)
        args, varargs, varkw, defaults = inspect.getargspec(f._wrapped_method)

        for arg, default in zip(args[1:], defaults):
            parser.add_argument("--"+arg, type=type(default), default=default)

        class kwarg_getter(object):
            def __init__(self, A):
                self.my_args = A
            def __call__(self, pargs):
                return {k:v for k,v in vars(pargs).iteritems()
                        if k in self.my_args}
        return kwarg_getter(args)
