import numpy as np
import h5py
from toolbase import MaraTool
from autolog import logmethod


class MaraCheckpointForceCalculator(MaraTool):
    @logmethod
    def __init__(self, filename):
        self._h5file = h5py.File(filename, 'r')

    @logmethod
    def magnetic_field(self):
        B = np.array([self._h5file['prim']['Bx'][:],
                      self._h5file['prim']['By'][:],
                      self._h5file['prim']['Bz'][:]])
        return B

    def get_ks(self, shape):
        """
        Get the wavenumber array, assuming all dimensions are length 1, while
        the number of cells along each axis may be unique
        """
        N0, N1, N2 = shape[1:]
        Ks = np.zeros(shape)
        Ks[0] = np.fft.fftfreq(N0)[:,None,None]*2*N0*np.pi
        Ks[1] = np.fft.fftfreq(N1)[None,:,None]*2*N1*np.pi
        Ks[2] = np.fft.fftfreq(N2)[None,None,:]*2*N2*np.pi
        return Ks

    @logmethod
    def magnetic_gradient_tensor(self, method='spectral', return_B=False):
        """
        Form the gradient tensor gradB[i,j] := dB_j/dx_i using either spectral
        or finite-difference method
        """
        Bx = self.magnetic_field()
        Ks = self.get_ks(Bx.shape)
        Bk = np.fft.fftn(Bx, axes=[1,2,3])
        N = Ks.shape[1:]
        d = five_point_deriv
        gradB = np.zeros((3,3) + N)
        for i in range(3):
            for j in range(3):
                if method == 'spectral':
                    gradB[i,j] = np.fft.ifftn(1.j * Ks[i] * Bk[j]).real
                elif method == 'finite-difference':
                    gradB[i,j] = d(Bx[j], i, h=1.0/N[i])
                else:
                    raise ValueError("bad argument value method='%s'" % method)
        if return_B:
            return gradB, Bx
        else:
            return gradB

    @logmethod
    def magnetic_pressure_gradient_force(self, method='spectral'):
        """ contract B_j with dB_j/dx_i; return -grad(B^2)/2 """
        gradB, B = self.magnetic_gradient_tensor(method=method, return_B=True)
        F = np.zeros_like(B)
        for i in range(3):
            for j in range(3):
                F[i] += B[j] * gradB[i,j]
        return F

    @logmethod
    def magnetic_tension(self, method='spectral'):
        """ contract B_i with dB_j/dx_i; return B dot grad(B) """
        gradB, B = self.magnetic_gradient_tensor(method=method, return_B=True)
        F = np.zeros_like(B)
        for i in range(3):
            for j in range(3):
                F[j] += B[i] * gradB[i,j]
        return F

    @logmethod
    def curlB(self):
        B = self.magnetic_field()
        N = B.shape[1:]
        d = lambda f, a: five_point_deriv(f, a, h=1.0/N[a])
        curl = np.zeros_like(B)
        curl[0] = d(B[0], 2) - d(B[2], 0)
        curl[1] = d(B[1], 0) - d(B[0], 1)
        curl[2] = d(B[2], 1) - d(B[1], 2)
        return curl


def five_point_deriv(f, axis=0, h=1.0):
    return (-1*np.roll(f, -2, axis) +
            +8*np.roll(f, -1, axis) + 
            -8*np.roll(f, +1, axis) + 
            +1*np.roll(f, +2, axis)) / (12.0 * h)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 5
    L = 3.0
    N = 256
    k = np.fft.fftfreq(N)*2*N*np.pi/L
    x = np.linspace(0.0, L*(1.0 - 1.0/N), N)
    y   = np.sin(2 * n * np.pi * x / L)
    yp1 = np.cos(2 * n * np.pi * x / L) * (2 * n * np.pi / L)
    yp2 = five_point_deriv(y, h=L/N)
    yp3 = np.fft.ifft(1.j * k * np.fft.fft(y)).real
    plt.plot(x, yp1, label="$y_1'(x)$")
    plt.plot(x, yp2, label="$y_2'(x)$")
    plt.plot(x, yp3, label="$y_3'(x)$")
    plt.legend()
    plt.show()
