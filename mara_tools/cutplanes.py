import numpy as np
import h5py
from toolbase import MaraTool
from autolog import logmethod

class MaraCheckpointCutplaneExtractor(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._chkpt = h5py.File(filename, 'r')

    @logmethod
    def plot_slice(self, field='rho', cmap='jet', axis=0, index=0,
                   sax=None, show=True):
        import matplotlib.pyplot as plt
        if axis == 0: imgdata = self._chkpt['prim'][field][index,:,:]
        if axis == 1: imgdata = self._chkpt['prim'][field][:,index,:]
        if axis == 2: imgdata = self._chkpt['prim'][field][:,:,index]

        if sax is None:
            fig = plt.figure(figsize=[10,10])
            sax = fig.add_subplot('111')
        else:
            fig = sax.get_figure()
        cax = sax.imshow(imgdata, origin='image', interpolation='nearest')
        sax.axes.get_xaxis().set_visible(False)
        sax.axes.get_yaxis().set_visible(False)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.colorbar(cax, shrink=0.85, pad=0.0, aspect=20, cmap=cmap,
                     orientation="horizontal")
        plt.setp(sax.get_xticklabels(), visible=False)
        plt.setp(sax.get_yticklabels(), visible=False)
        if show:
            plt.show()

    @logmethod
    def plot_lic(self, index=0, texture=None):
        import matplotlib.pyplot as plt
        from lic import lic_internal

        v1 = self._chkpt['prim']['By'][index,:,:]
        v2 = self._chkpt['prim']['Bz'][index,:,:]
        N = v1.shape

        vectors = np.zeros(N + (2,), dtype=np.float32)
        vectors[:,:,0] = v1
        vectors[:,:,1] = v2

        if texture is None:
            texture = np.random.rand(N[0],N[1]).astype(np.float32)

        kernellen = 32
        kernel = np.sin(np.arange(kernellen)*np.pi/kernellen).astype(np.float32)
        image = lic_internal.line_integral_convolution(vectors, texture, kernel)

        plt.imshow(image, cmap='bone', interpolation='nearest')
        plt.show()

        return texture
