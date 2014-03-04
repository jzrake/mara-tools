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
    def __init__(self, size, rms=1.0, Pofk=None, random_state=None):
        if random_state is None:
            random_state = np.random.get_state()
        else:
            np.random.set_state(random_state)
        Ax = np.random.uniform(-0.5, 0.5, [3] + [size]*3)
        Ks = get_ks(Ax.shape)
        Ak = np.fft.fftn(Ax, axes=[1,2,3])
        Ak[:,size/2,:,:] = 0.0 # zero-out the Nyquist frequencies
        Ak[:,:,size/2,:] = 0.0
        Ak[:,:,:,size/2] = 0.0
        K2 = Ks[0]**2 + Ks[1]**2 + Ks[2]**2
        K2[0,0,0] = 1.0 # prevent divide-by-zero
        if Pofk is None: Pofk = lambda k: 1.0
        Pk = Pofk(K2**0.5)
        Ak *= (Pk / K2)**0.5
        self._Ak = Ak
        self._Ks = Ks
        self._K2 = K2
        self._rms = rms
        self._random_state = random_state

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
    def get_field(self, zeta=None, kcomp=None, force_free=None,
                  perturbation=0.0, mean_field=[0, 0, 0]):
        """
        zeta:
        -----

        Helmoltz decomposition parameter, get a field which is zeta part
        solenoidal and 1 - zeta part dilatational.


        kcomp:
        ------

        If float, multiply power spectrum by power law with index kcomp.


        force_free:
        -----------

        If float, generate a field satisfying the linear force-free condition
        del^2 B + alpha^2 B = 0, such that magnetic tension is balanced by the
        magnetic pressure gradient. This is equivalent to the power spectrum
        having compact support at |k| = alpha, where alpha has the value of the
        force_free parameter. **CURRENTLY BROKEN**


        perturbation:
        -------------

        If in force-free mode and perturbation is not zero, then instead of
        zero-ing out the off-shell modes, just suppress them by this factor.
        """
        if zeta is not None:
            Dk, Sk = self.helmholtz()
            Ak = zeta * Sk + (1.0 - zeta) * Dk
        else:
            Ak = self._Ak

        # compensate by k^kcomp if asked
        if kcomp is not None:
            Ak *= self._K2**(kcomp/2.0)

        # zero out all the Fourier amplitudes not on the force-free wavenumber
        if force_free is not None:
            raise NotImplementedError("force-free field creation is "
                                      "currently broken")
            a2 = self._K2.flat[np.argmin(abs(self._K2 - force_free**2))]
            Ak[:,self._K2 != a2] *= perturbation

        Ax = np.fft.ifftn(Ak, axes=[1,2,3]).real
        Pxs = Ax[0]**2 + Ax[1]**2 + Ax[2]**2
        A0 = np.array(mean_field)[:,None,None,None]
        return A0 + self._rms * Ax / Pxs.mean()**0.5

    @logmethod
    def get_toth_potential_field(self, force_free=None, mean_field=[0, 0, 0]):
        """
        Return a Gaussian random field which is divergenceless according to the
        corner-centered stencil of Toth (2000), by treating the Fourier
        amplitudes as a vector potential
        """
        if force_free is not None:
            raise NotImplementedError("cannot generate force-free fields "
                                      "that are Toth divergenceless")

        Ak = self.get_field(kcomp=-1.0)

        def ct(a, sx, sy):
            F = -0.5*(a + sx(a,-1))
            G = +0.5*(a + sy(a,-1))
            fxby = 2*F+sy(F,-1)+sy(F,+1)-G-sx(G,-1)-sy(G,+1)-sx(sy(G,+1),-1)
            fybx = 2*G+sx(G,-1)+sx(G,+1)-F-sy(F,-1)-sx(F,+1)-sy(sx(F,+1),-1)
            return fxby, fybx

        FxBy, FyBx = ct(Ak[2],
                        lambda f,i: np.roll(f, i, axis=0),
                        lambda f,i: np.roll(f, i, axis=1))
        FyBz, FzBy = ct(Ak[0],
                        lambda f,i: np.roll(f, i, axis=1),
                        lambda f,i: np.roll(f, i, axis=2))
        FzBx, FxBz = ct(Ak[1],
                        lambda f,i: np.roll(f, i, axis=2),
                        lambda f,i: np.roll(f, i, axis=0))

        B = np.zeros(Ak.shape)
        B[0] = FyBx - np.roll(FyBx, 1, axis=1) + FzBx - np.roll(FzBx, 1, axis=2)
        B[1] = FzBy - np.roll(FzBy, 1, axis=2) + FxBy - np.roll(FxBy, 1, axis=0)
        B[2] = FxBz - np.roll(FxBz, 1, axis=0) + FyBz - np.roll(FyBz, 1, axis=1)
        Pxs = B[0]**2 + B[1]**2 + B[2]**2
        B0 = np.array(mean_field)[:,None,None,None]
        return B0 + self._rms * B / Pxs.mean()**0.5

    @logmethod
    def power_spectrum(self, bins=128, zeta=None):
        if zeta is not None:
            Dk, Sk = self.helmholtz()
            Ak = zeta * Sk + (1.0 - zeta) * Dk
        else:
            Ak = self._Akg
        return power_spectrum(None, Ak=Ak, Ks=self._Ks, bins=bins)

    @property
    def random_state(self):
        return self._random_state


def get_ks(shape):
    """
    Get the wavenumber array, assuming all dimensions are length 1, while the
    number of cells along each axis may be unique. NOTE: These wavenumbers give
    derivatives that small by a factor of 2 pi / L where L is the domain
    size. They are convenient in that k=1.0 means the domain size, k=10.0 is
    1/10 the domain size and so on.
    """
    N0, N1, N2 = shape[1:]
    Ks = np.zeros(shape)
    Ks[0] = np.fft.fftfreq(N0)[:,None,None]*2*N0
    Ks[1] = np.fft.fftfreq(N1)[None,:,None]*2*N1
    Ks[2] = np.fft.fftfreq(N2)[None,None,:]*2*N2
    return Ks


def divergence(f):
    """
    Compute the divergence of the vector field f according to the
    corner-centered stencil of Toth (2000)
    """
    roll = np.roll
    def R(A, i, j, k, axis):
        i,j,k = [(i,j,k), (k,i,j), (j,k,i)][axis]
        return roll(roll(roll(A, i, axis=0), j, axis=1), k, axis=2)
    div = [((R(f[i],1,0,0,i) + R(f[i],1,1,0,i) + R(f[i],1,0,1,i) + R(f[i],1,1,1,i)) -
            (R(f[i],0,0,0,i) + R(f[i],0,1,0,i) + R(f[i],0,0,1,i) + R(f[i],0,1,1,i)))
           for i in range(3)]
    return (div[0] + div[1] + div[2]) / 4.0


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
        Ks = get_ks(s)
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
    field = GaussianRandomVectorField3d(64, rms=4.0, Pofk=lambda k: k**p)
    #A = field.get_field(zeta=1.0)
    A = field.get_toth_potential_field()
    D = divergence(A)
    print "Toth div's are:", D.std(), D.min(), D.max()
    print "field RMS is:", (A[0]*A[0] + A[1]*A[1] + A[2]*A[2]).mean()**0.5
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
