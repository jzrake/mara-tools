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


def fig_cascade_pspec_evolve():
    import matplotlib.pyplot as plt

    def ax_params(**kwargs):
        params = {
            'MaraReductionsReader': {
                'kcut': [0.0, 128.0],
                'xlim': [2.0/1.25, 128.0*1.25],
                'alpha': 1.0,
                'comp': 0.0,
                }
            }
        params['MaraReductionsReader'].update(kwargs)
        return params

    def trim_yaxis(ax):
        yl = ax.get_ylim()
        ax.set_ylim(yl[0]*1.001, yl[1]/1.001)

    ax1_params = ax_params(ylabel=r'$P_B(k)$', xlabel='')
    ax2_params = ax_params(ylabel=r'$P_{v,s}(k)$', xlabel='')
    ax3_params = ax_params(ylabel=r'$P_{v,d}(k)$', xlabel=r'$k L/2\pi$')

    y0 = 0.05
    fig = plt.figure(figsize=[6,12])
    ax1 = fig.add_subplot('311')
    ax2 = fig.add_subplot('312')
    ax3 = fig.add_subplot('313')

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    pargs = parser.parse_args()
    reduc = reductions.MaraReductionsReader(pargs.filename)

    skip = 100
    reduc.set_show_action('hold')
    reduc.set_plot_axes(ax1)
    reduc.update_user_params(ax1_params)
    reduc.plot_power_spectra(which=['magnetic-solenoidal'],
                             tmin=0.00,
                             skip=skip,
                             nolegend=True,
                             cmap='gnuplot')
    reduc.set_plot_axes(ax2)
    reduc.update_user_params(ax2_params)
    reduc.plot_power_spectra(which=['velocity-solenoidal'],
                             tmin=0.01,
                             skip=skip,
                             nolegend=True,
                             cmap='gnuplot')
    reduc.set_plot_axes(ax3)
    reduc.update_user_params(ax3_params)
    reduc.plot_power_spectra(which=['velocity-dilatational'],
                             tmin=0.01,
                             skip=skip,
                             nolegend=True,
                             cmap='gnuplot')

    ax1.axes.get_xaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)

    trim_yaxis(ax1)
    trim_yaxis(ax2)
    trim_yaxis(ax3)

    x = np.linspace(2.0, 128.0, 1000)
    y1 = 5e-3 * x**-2.0
    y2 = 5e-6 * x**-0.5
    ax1.loglog(x, y1, '--', c='k', lw=2.0)
    ax2.loglog(x, y2, '--', c='k', lw=2.0)

    for ax, letter, text in zip([ax1, ax2, ax3],
                        "abc",
                        [r"$P_B(k) \propto k^{-2}$",
                         r"$P_{v,s}(k) \propto k^{-1/2}$",
                         ""]):
        ax.text(0.1, 0.4, text, transform=ax.transAxes, fontsize=16)
        ax.text(0.05, 0.05, r"$(%c)$"%letter, transform=ax.transAxes, fontsize=16)

    fig.subplots_adjust(left=0.15, bottom=0.05, right=0.98, top=0.98, hspace=0.0)
    plt.show()


def fig_cascade_peak_power_evolve():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[8,10])
    ax1 = fig.add_subplot('211')
    ax2 = fig.add_subplot('212')

    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("checkpoint")
    pargs = parser.parse_args()
    reduc = reductions.MaraReductionsReader(pargs.filename)
    log = chkptlog.MaraCheckpointLoggedData(pargs.checkpoint)

    def fit_loglog(X, Y, x0, x1, ax, voffs=1.1, *args, **kwargs):
        x = X[(X<x1)*(X>x0)]
        y = Y[(X<x1)*(X>x0)]
        m = np.log10(y[-1]/y[0])/np.log10(x[-1]/x[0])
        z = y[0] * (x/x[0])**m
        ax.loglog(x, voffs * z, *args, **kwargs)
        return m, z

    skip = 1

    t, k, P = reduc.rms_wavenumber_all('magnetic-solenoidal', skip=skip)
    m, z = fit_loglog(t, 1/k, 1.0, 100.0, ax2, ls='--', c='k')
    ax2.loglog(t, 1/k, lw=1.5, c='b',
               label=r'magnetic $\lambda_{RMS} \propto t^{%3.2f}$'%m)

    t, k, P = reduc.rms_wavenumber_all('velocity-solenoidal', skip=skip)
    m, z = fit_loglog(t, 1/k, 1.0, 100.0, ax2, ls='--', c='k')
    ax2.loglog(t, 1/k, lw=1.5, c='g',
               label=r'solenoidal velocity $\lambda_{RMS} \propto t^{%3.2f}$'%m)

    t, k, P = reduc.rms_wavenumber_all('velocity-dilatational', skip=skip)
    m, z = fit_loglog(t, 1/k, 1.0, 100.0, ax2, ls='--', c='k')
    ax2.loglog(t, 1/k, lw=1.5, c='r',
               label=r'dilatational velocity $\lambda_{RMS} \propto t^{%3.2f}$'%m)

    m_mag, z_mag = fit_loglog(log.time, log.mag, 1.0, 100.0, ax1, voffs=2, ls='--', c='k')
    m_kin, z_kin = fit_loglog(log.time, log.kin, 1.0, 100.0, ax1, voffs=2, ls='--', c='k')

    ax1.loglog(log.time, log.mag, c='b', lw=1.5, label='magnetic $\propto t^{%3.2f}$'%m_mag)
    ax1.loglog(log.time, log.kin, c='g', lw=1.5, label='kinetic $\propto t^{%3.2f}$'%m_kin)
    ax1.axes.get_xaxis().set_visible(False)

    ax2.set_xlabel(r'$t$', fontsize=18)
    ax2.set_ylabel(r'$\lambda_{RMS}$', fontsize=18)
    ax1.set_ylabel(r'energy', fontsize=18)
    ax1.set_xlim(min(t)/1.1, max(t)*1.1)
    ax2.set_xlim(min(t)/1.1, max(t)*1.1)
    ax1.set_ylim(9e-7, 8e-1)
    ax2.set_ylim(1e-2, 2e-1)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    fig.subplots_adjust(hspace=0)
    plt.show()


def fig_cascade_resolution_mag_decay():
    import matplotlib.pyplot as plt
    import json

    fig = plt.figure(figsize=[10,8])
    ax1 = fig.add_subplot('111')

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs=4)
    pargs = parser.parse_args()

    for filename in pargs.filenames:
        chkpt = h5py.File(filename, 'r')
        runargs = json.loads(chkpt['runargs'].value)
        N = int(runargs['N'])
        chkpt.close()

        log = chkptlog.MaraCheckpointLoggedData(filename)
        ax1.loglog(log.time, log.mag, lw=1.5, label='magnetic'+' '+filename)
        if N == 512:
            ax1.text(log.time[1]/2, 0.5, r'$%d^3$'%N, fontsize=18)
        else:
            ax1.text(log.time[1]/1, 0.5, r'$%d^3$'%N, fontsize=18)

    ax1.set_xlabel('time (Alfven crossings)', fontsize=18)
    ax1.set_ylabel('magnetic energy', fontsize=18)
    plt.show()
