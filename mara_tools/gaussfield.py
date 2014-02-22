import numpy as np
from autolog import logmethod

class GaussianRandomVectorField3d(object):
    """
    Generates a Gaussian random vector field whose spatial realization is
    accomplished by a Fourier transform instead of summation of the trig
    series. It cannot be perturbed stochastically like the StochasticVectorField
    classes.
    """
    @logmethod
    def __init__(self, size, rms=1.0, Pofk=None):
        Ax = np.random.uniform(-0.5, 0.5, [3] + [size]*3)
        Ks = np.zeros(Ax.shape, dtype=float)
        Ak = np.fft.fftn(Ax, axes=[1,2,3])
        Ak[:,size/2,:,:] = 0.0
        Ak[:,:,size/2,:] = 0.0
        Ak[:,:,:,size/2] = 0.0
        Ks[0] = np.fft.fftfreq(size)[:,None,None]
        Ks[1] = np.fft.fftfreq(size)[None,:,None]
        Ks[2] = np.fft.fftfreq(size)[None,None,:]
        K2 = np.abs(Ks[0])**2 + np.abs(Ks[1])**2 + np.abs(Ks[2])**2
        K2[0,0,0] = 1.0 # prevent divide-by-zero
        if Pofk is None: Pofk = lambda k: 1.0
        Pk = Pofk(K2**0.5)
        Ak *= (Pk / K2)**0.5
        self._Ak = Ak
        self._Ks = Ks
        self._K2 = K2
        self._rms = rms

    @logmethod
    def root_mean_square(self):
        Ax = self.get_field()
        Pxs = np.abs(Ax[0])**2 + np.abs(Ax[1])**2 + np.abs(Ax[2])**2
        return Pxs.mean()**0.5

    def helmholtz(self):
        """
        Return the dilatational and solenoidals parts of the Fourier
        representation of the field
        """
        return helmholtz(self._Ak, self._Ks)

    @logmethod
    def get_field(self, zeta=None):
        if zeta is not None:
            Dk, Sk = self.helmholtz()
            Cx = np.fft.ifftn(zeta * Sk + (1.0 - zeta) * Dk, axes=[1,2,3])
        else:
            Cx = np.fft.ifftn(self._Ak, axes=[1,2,3])
        Ax = Cx.real
        Pxs = Ax[0]**2 + Ax[1]**2 + Ax[2]**2
        return self._rms * Ax / Pxs.mean()**0.5

    @logmethod
    def power_spectrum(self, bins=128, zeta=None):
        if zeta is not None:
            Dk, Sk = self.helmholtz()
            Ak = zeta * Sk + (1.0 - zeta) * Dk
        else:
            Ak = self._Ak
        return power_spectrum(None, Ak=Ak, Ks=self._Ks, bins=bins)



def helmholtz(Ak, Ks):
    """
    Return the dilatational and solenoidals parts of the Fourier
    representation of the field
    """
    K2 = Ks[0]**2 + Ks[1]**2 + Ks[2]**2; K2[0,0,0] = 1.0
    Kh = Ks / K2**0.5
    Dk = (Ak[0]*Kh[0] + Ak[1]*Kh[1] + Ak[2]*Kh[2]) * Kh
    Sk = Ak - Dk
    return Dk, Sk


def power_spectrum(Ax, Ak=None, Ks=None, bins=128):
    if Ks is None:
        s = Ax.shape if Ax is not None else Ak.shape
        Ks = np.zeros(s, dtype=float)
        Ks[0] = np.fft.fftfreq(s[1])[:,None,None]
        Ks[1] = np.fft.fftfreq(s[2])[None,:,None]
        Ks[2] = np.fft.fftfreq(s[3])[None,None,:]
    if Ak is None:
        Ak = np.fft.fftn(Ax, axes=[1,2,3])

    K2 = Ks[0]**2 + Ks[1]**2 + Ks[2]**2; K2[0,0,0] = 1.0
    Dk, Sk = helmholtz(Ak, Ks)

    PsD = np.abs(Dk[0])**2 + np.abs(Dk[1])**2 + np.abs(Dk[2])**2
    PsS = np.abs(Sk[0])**2 + np.abs(Sk[1])**2 + np.abs(Sk[2])**2

    valsD, bins = np.histogram(K2**0.5, weights=PsD, bins=bins)
    valsS, bins = np.histogram(K2**0.5, weights=PsS, bins=bins)

    dPD = valsD
    dPS = valsS
    dk = 1.0 * (bins[1:] - bins[:-1])
    k0 = 0.5 * (bins[1:] + bins[:-1])

    return k0, dPD/dk, dPS/dk



if __name__ == "__main__":
    """
    Example usage: create a Kolmogorov Gaussian vector field
    """
    import matplotlib.pyplot as plt
    p = -5./3.
    field = GaussianRandomVectorField3d(128, rms=4.0, Pofk=lambda k: k**p)
    A = field.get_field(zeta=1.0)
    #k, PD, PS = field.power_spectrum(zeta=1.0)
    k, PD, PS = power_spectrum(A)
    i = np.argmax(PS)

    plt.figure()
    plt.loglog(k, PD, label='dilatational')
    plt.loglog(k, PS, label='solenoidal')
    plt.loglog(k, PS[i]*(k/k[i])**p)
    plt.legend(loc='best')
    plt.show()

    plt.figure()
    plt.imshow(A[0,:,:,0], interpolation='nearest')
    plt.colorbar()
    plt.show()
