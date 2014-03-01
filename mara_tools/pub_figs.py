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
             {'A': 1.0, 'B': 2.0, 'C': 4.0, 'D': 8.0}[c])
            for c in "ABCD"]

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
    plt.title('Comparison of decaying trans-sonic turbulence '
              'with supersonic boosts (WENO-5)')
    plt.show()

