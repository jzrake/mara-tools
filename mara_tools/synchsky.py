import itertools
import numpy as np
import h5py
from autolog import logmethod


"""
Synchrotron visualization of the HDF5 data, with loading the data one row at a
time to allow for processing large files.
"""


class StokesMaraCheckpointFile(object):

    @logmethod
    def __init__(self, filename):
        self.h5file = h5py.File(filename, 'r')
        if 'prim' not in self.h5file:
            raise RuntimeError("does not look like a Mara checkpoint")

    def stokes_line(self, pg, b1, b2):
        """
        Return the Stokes vector components I, Q, U, and polarization fraction
        for a given pixel by integrating along the one-dimensional arrays for
        pressure `pg` and transverse magnetic field `b1` and `b2`
        """
        alpha=0.5
        maxpol=(alpha+1.0)/(alpha+5.0/3.0)
        bs=(b1**2.0+b2**2.0)**(0.5*(alpha-1.0))
        iem=pg*bs*(b1**2.0+b2**2.0)
        qem=maxpol*pg*bs*(b1**2.0-b2**2.0)
        uem=maxpol*pg*bs*(2.0*b1*b2)
        si=np.sum(iem)
        sq=np.sum(qem)
        su=np.sum(uem)
        fp=np.sqrt(sq**2.0+su**2.0)/si
        return si, sq, su, fp

    @logmethod
    def stokes_image(self, axis=0):
        """
        Return the Stokes vector components I, Q, U, and polarization fraction
        as four images
        """
        S = self.h5file['prim']['pre'].shape
        ni, nj = [(S[1],S[2]), (S[2],S[0]), (S[0],S[1])][axis]
        I = np.zeros([ni, nj])
        Q = np.zeros([ni, nj])
        U = np.zeros([ni, nj])
        P = np.zeros([ni, nj])

        # ----------------------------------------
        # integrate along the x-axis
        # ----------------------------------------
        if axis == 0:
            for k in range(S[2]):
                print "loading plane %d/%d" % (k, S[2])
                pg = self.h5file['prim']['pre'][:,:,k]
                b1 = self.h5file['prim']['By' ][:,:,k]
                b2 = self.h5file['prim']['Bz' ][:,:,k]
                for j in range(S[1]):
                    si,sq,su,fp = self.stokes_line(pg[:,j], b1[:,j], b2[:,j])
                    I[j,k] = si
                    Q[j,k] = sq
                    U[j,k] = su
                    P[j,k] = fp

        # ----------------------------------------
        # integrate along the y-axis
        # ----------------------------------------
        if axis == 1:
            for i in range(S[0]):
                print "loading plane %d/%d" % (i, S[0])
                pg = self.h5file['prim']['pre'][i,:,:]
                b1 = self.h5file['prim']['Bz' ][i,:,:]
                b2 = self.h5file['prim']['Bx' ][i,:,:]
                for k in range(S[2]):
                    si,sq,su,fp = self.stokes_line(pg[:,k], b1[:,k], b2[:,k])
                    I[k,i] = si
                    Q[k,i] = sq
                    U[k,i] = su
                    P[k,i] = fp

        # ----------------------------------------
        # integrate along the z-axis
        # ----------------------------------------
        if axis == 2:
            for j in range(S[1]):
                print "loading plane %d/%d" % (j, S[1])
                pg = self.h5file['prim']['pre'][:,j,:]
                b1 = self.h5file['prim']['Bx' ][:,j,:]
                b2 = self.h5file['prim']['By' ][:,j,:]
                for i in range(S[0]):
                    si,sq,su,fp = self.stokes_line(pg[i,:], b1[i,:], b2[i,:])
                    I[i,j] = si
                    Q[i,j] = sq
                    U[i,j] = su
                    P[i,j] = fp

        return StokesVectorsImage(I, Q, U, P, axis, self.h5file.filename)


class StokesVectorsImage(object):

    @logmethod
    def __init__(self, I=None, Q=None, U=None, P=None, axis=None, filename=None):
        self.I = I
        self.Q = Q
        self.U = U
        self.P = P
        self.axis = axis
        self.filename = str(filename)

    @logmethod
    def load(self, filename):
        h5file = h5py.File(filename, 'r')
        self.I = h5file['stokes-I'][:]
        self.Q = h5file['stokes-Q'][:]
        self.U = h5file['stokes-U'][:]
        self.P = h5file['polarization'][:]
        self.axis = h5file.attrs['axis']
        self.filename_orig = h5file.attrs['filename']
        self.filename_synch = filename

    @logmethod
    def save(self, outfile):
        outf = h5py.File(outfile, 'w')
        outf['stokes-I'] = self.I
        outf['stokes-Q'] = self.Q
        outf['stokes-U'] = self.U
        outf['polarization'] = self.P
        outf.attrs['filename'] = self.filename
        outf.attrs['axis'] = self.axis
        outf.close()

    @logmethod
    def make_figures1(self):
        import matplotlib.pyplot as plt
        si, sq, su, fp = self.I, self.Q, self.U, self.P
        plt.figure(figsize=(15,3))
        plt.subplot(1,5,1)
        plt.imshow(si,origin='lower')
        plt.subplot(1,5,2)
        plt.imshow(sq,origin='lower')
        plt.subplot(1,5,3)
        plt.imshow(su,origin='lower')
        plt.subplot(1,5,4)
        plt.imshow(fp,origin='lower')
        plt.subplot(1,5,5)
        plt.hist(fp.flatten(),100,range=(0.0,1.0),histtype='step',normed=1)
        plt.show()   

    @logmethod
    def make_figures2(self, cmap='bone', hardcopy=False):
        import matplotlib.pyplot as plt
        si, sq, su, fp = self.I, self.Q, self.U, self.P

        for s, short, t in zip([si, sq, su, fp],
                        ['si', 'sq', 'su', 'fp'],
                        ["Stokes-I: axis %d" % self.axis,
                         "Stokes-Q: axis %d" % self.axis,
                         "Stokes-U: axis %d" % self.axis,
                         "Polarization fraction: axis %d" % self.axis]):

            fig = plt.figure(figsize=(10,12))
            sax = plt.gca()
            cax = sax.imshow(s, origin='lower', cmap=cmap)
            sax.set_title(t)
            fig.colorbar(cax, ax=sax, shrink=1.0, pad=0.0, aspect=40,
                         cmap=cmap, orientation="horizontal")
            fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
            plt.setp(sax.get_xticklabels(), visible=False)
            plt.setp(sax.get_yticklabels(), visible=False)
            sax.axes.get_xaxis().set_visible(False)
            sax.axes.get_yaxis().set_visible(False)

            if hardcopy:
                plt.savefig(self.filename_synch.replace(
                        '.h5', '-%s.png'%short))
                plt.clf()

        if not hardcopy:
            plt.show()   
