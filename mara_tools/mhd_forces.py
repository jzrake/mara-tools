import numpy as np
import h5py
from toolbase import MaraTool
from autolog import logmethod

class MaraCheckpointForceCalculator(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._h5file = h5py.File(filename, 'r')

    @logmethod
    def magnetic_pressure_gradient(self):
        Bx = np.array([self._h5file['prim']['Bx'][:],
                       self._h5file['prim']['By'][:],
                       self._h5file['prim']['Bz'][:]])

        Ks = np.zeros(Bx.shape, dtype=float)

        Ks[0] = np.fft.fftfreq(Ks.shape[1])[:,None,None]
        Ks[1] = np.fft.fftfreq(Ks.shape[2])[None,:,None]
        Ks[2] = np.fft.fftfreq(Ks.shape[3])[None,None,:]

        Pbx = 0.5*(Bx[0]**2 + Bx[1]**2 + Bx[2]**2)
        Pbk = np.fft.fftn(Pbx)
        Fbk = 1.j * Ks * Pbk
        Fbx = np.fft.ifftn(Fbk, axes=[1,2,3]).real

        return Fbx


    @logmethod
    def magnetic_tension(self):
        Bx = np.array([self._h5file['prim']['Bx'][:],
                       self._h5file['prim']['By'][:],
                       self._h5file['prim']['Bz'][:]])
        Bk = np.fft.fftn(Bx, axes=[1,2,3])

        Ks = np.zeros(list(Bx.shape), dtype=float)
        Ks[0] = np.fft.fftfreq(Ks.shape[1])[:,None,None]
        Ks[1] = np.fft.fftfreq(Ks.shape[2])[None,:,None]
        Ks[2] = np.fft.fftfreq(Ks.shape[3])[None,None,:]    

        Fbx = np.zeros_like(Bx)

        for i in range(3):
            for j in range(3):
                gradBk_ij = -1.j * Ks[i] * Bk[j]
                gradBx_ij = np.fft.ifftn(gradBk_ij).real
                Fbx[i] += Bx[j] * gradBx_ij

        return Fbx
