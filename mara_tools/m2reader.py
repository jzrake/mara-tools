import numpy as np
import h5py
from toolbase import MaraTool
from autolog import logmethod


class M2CheckpointReader(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._h5file = h5py.File(filename, 'r')

    def make_1d_plots(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import AxesGrid

        def trim_axis(ax):
            xl = ax.get_xlim()
            yl = ax.get_ylim()
            ax.set_xlim(xl[0]+0.001, xl[1]-0.001)
            ax.set_ylim(yl[0]+0.001, yl[1]-0.001)

        fig = plt.figure(1, (14, 10))
        fig.subplots_adjust(left=0.05, right=0.98)
        grid = AxesGrid(fig, 111,
                        nrows_ncols = (2, 4),
                        axes_pad = 0.05,
                        share_all=True,
                        add_all=True,
                        #direction='column',
                        label_mode='L')

        dsets = ['p','B1','B2','B3','d','v1','v2','v3']
        ymin, ymax = [], []
        for i, dset in enumerate(dsets):
            y = self._h5file['prim'][dset].value
            ymin.append(y.min())
            ymax.append(y.max())

        spread = max(ymax) - min(ymin)

        for i, dset in enumerate(dsets):
            y = self._h5file['prim'][dset].value
            x = np.linspace(0.0, spread, y.size)
            grid[i].plot(x, y, c='k', lw=2.0, marker='o', mfc='none')
            grid[i].text(0.1, 0.1, dset, transform=grid[i].transAxes, fontsize=16)
            grid[i].set_xlim(0.0, spread)

        trim_axis(grid[0])
        self.show()

    def make_2d_plots(self, field):
        import matplotlib.pyplot as plt

        #cmap = 'cubehelix'
        cmap = 'jet'
        imgdata = self._h5file['prim'][field].value.T
        scale_std = False

        if scale_std:
            dz = 3 * imgdata.std()
            vmin = -dz
            vmax = +dz
        else:
            vmin = imgdata.min()
            vmax = imgdata.max()

        sax = self.get_plot_axes()
        fig = sax.get_figure()

        cax = sax.imshow(imgdata, cmap=cmap, origin='image',
                         interpolation='nearest', vmin=vmin, vmax=vmax)

        fig.colorbar(cax, ax=sax, shrink=0.85, pad=0.0, aspect=20,
                     cmap=cmap, orientation="horizontal")
        fig.suptitle(self._h5file.filename)

        sax.axes.get_xaxis().set_visible(False)
        sax.axes.get_yaxis().set_visible(False)
        plt.setp(sax.get_xticklabels(), visible=False)
        plt.setp(sax.get_yticklabels(), visible=False)

        self.show()
