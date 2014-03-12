import sys
import os
import argparse
import pickle
import numpy as np
import h5py
from mara_tools import gaussfield
from mara_tools import chkptlog
from mara_tools import synchsky
from mara_tools import genchkpt
from mara_tools import cutplanes
from mara_tools import reductions
from mara_tools import mhd_forces


def fig_power_spectrum_mach_boost():
    import matplotlib.pyplot as plt
    cls = reductions.MaraReductionsReader
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--time", type=float, help="spectrum near time",
                        default=1.0)
    parser.add_argument("--which", type=str, default="velocity-solenoidal")
    parser.add_argument("--data-dir", type=str, default='data')
    pargs = parser.parse_args()

    runs = [(os.path.join(pargs.data_dir, "hdec.%c-256.spec.h5"%c),
             {'E': 1.0, 'F': 2.0, 'G': 4.0, 'H': 8.0}[c])
            for c in "EFGH"]

    plt.figure(figsize=[12,10])
    for filename, mach in runs:
        reduc = cls(filename)
        k, P = reduc.plot_single_power_spectrum(pargs.time, pargs.which,
                                                comp=5./3., lw=2.0,
                                                label="Mach %2.1f"%mach)
    plt.loglog(k, 0.1*k**0.0, label=r'$k^{-5/3}$')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$k^{5/3} dP/dk$')
    plt.legend(loc='best')
    plt.title('Comparison of decaying sub-sonic turbulence '
              'with supersonic boosts (WENO-5)')
    plt.show()


def fig_cutplane_pspec_hybrid():
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+', type=str)
    parser.add_argument("--spec-file", type=str, required=True)
    pargs = parser.parse_args()
    pspec = reductions.MaraReductionsReader(pargs.spec_file)

    hp = 0.05
    vp = 0.12

    pargs.filenames.sort()
    cl = chkptlog.MaraCheckpointLoggedData(pargs.filenames[-1])

    xfmt = ticker.ScalarFormatter()

    for filename in pargs.filenames:
        fig = plt.figure(figsize=[16,10])
        ax1 = fig.add_axes([0.00+hp/2, 0.0+vp/2, 0.66-hp, 1.0-vp*0.8])
        ax2 = fig.add_axes([0.66+hp/2, 0.5+vp/2, 0.33-hp, 0.5-vp*0.8], axisbg=[0.9,0.9,1.0])
        ax3 = fig.add_axes([0.66+hp/2, 0.0+vp/2, 0.33-hp, 0.5-vp*0.8], axisbg=[0.9,0.9,1.0])

        cp = cutplanes.MaraCheckpointCutplaneExtractor(filename)
        cp.plot_slice(field='Bx', plot_axis=ax1, cmap='bone', noshow=True)
        #cp.plot_lic(plot_axis=ax1, noshow=True)
        #cp.plot_streamlines(plot_axis=ax1, noshow=True)
        cl.plot_fields(['mag', 'kin'], plot_axis=ax3, noshow=True, lw=3.0)

        time = cp.get_status('CurrentTime')
        pspec.plot_single_power_spectrum(time=time, which='magnetic-solenoidal',
                                         plot_axis=ax2, lw=3.0, label='magnetic energy')
        pspec.plot_single_power_spectrum(time=time, which='velocity-solenoidal',
                                         plot_axis=ax2, lw=3.0, label='kinetic energy')
        ax2.legend(loc='lower left')
        ax2.set_xlim(2, 128)
        ax2.set_ylim(1e-12, 1e-1)
        ax2.set_xlabel('inverse scale', fontsize=18)
        ax2.set_ylabel(r'$P(k)$', fontsize=18)

        ax3.axvline(time, ls='-', lw=18.0, c='k', alpha=0.25)
        ax3.set_xlim(0.0, cl.time[-1])
        ax3.set_xlabel('time (Alfven crossing)', fontsize=18)
        ax3.set_ylabel(r'energy', fontsize=18)

        ax2.xaxis.set_major_formatter(xfmt)
        ax3.xaxis.set_major_formatter(xfmt)

        #plt.show()
        plt.savefig(filename.replace('.h5', '.png'))
        plt.clf()


def fig_cutplane_four_panel():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs=4, type=str)
    pargs = parser.parse_args()

    pargs.filenames.sort()

    fig = plt.figure(figsize=[12,4])
    ax1 = fig.add_axes([0.00, 0.0, 0.25, 1.0])
    ax2 = fig.add_axes([0.25, 0.0, 0.25, 1.0])
    ax3 = fig.add_axes([0.50, 0.0, 0.25, 1.0])
    ax4 = fig.add_axes([0.75, 0.0, 0.25, 1.0])

    for ax, filename in zip([ax1, ax2, ax3, ax4], pargs.filenames):
        cp = cutplanes.MaraCheckpointCutplaneExtractor(filename)
        cp.plot_slice(field='Bx', plot_axis=ax, cmap='bone', noshow=True)

    plt.show()

