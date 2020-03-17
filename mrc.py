# uncompyle6 version 3.6.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)]
# Embedded file name: /home/mab/Projects/cryosparc-package/cryosparc-compute/dataio/mrc.py
# Compiled at: 2016-12-27 15:32:21
import numpy as n

def readMRC(fname, inc_header=False):
    hdr = readMRCheader(fname)
    nx = hdr['nx']
    ny = hdr['ny']
    nz = hdr['nz']
    datatype = hdr['datatype']
    with open(fname) as (f):
        f.seek(1024)
        if datatype == 0:
            data = n.reshape(n.fromfile(f, dtype='int8', count=nx * ny * nz), (nx, ny, nz), order='F')
        elif datatype == 1:
            data = n.reshape(n.fromfile(f, dtype='int16', count=nx * ny * nz), (nx, ny, nz), order='F')
        elif datatype == 2:
            data = n.reshape(n.fromfile(f, dtype='float32', count=nx * ny * nz), (nx, ny, nz), order='F')
        else:
            assert False, ('Unsupported MRC datatype: {0}').format(datatype)
    if inc_header:
        return (data, hdr)
    else:
        return data


def writeMRC(fname, data, psz=1):
    """ Writes a MRC file. The header will be blank except for nx,ny,nz,datatype=2 for float32. 
    data should be (nx,ny,nz), and will be written in Fortran order as MRC requires."""
    header = n.zeros(256, dtype=n.int32)
    header_f = header.view(n.float32)
    header[:3] = data.shape
    if data.dtype == n.int8:
        header[3] = 0
    elif data.dtype == n.int16:
        header[3] = 1
    elif data.dtype == n.float32:
        header[3] = 2
    else:
        assert False, ('Unsupported MRC datatype: {0}').format(data.dtype)
    header[7:10] = data.shape
    header_f[10:13] = [ psz * i for i in data.shape ]
    header_f[13:16] = 90.0
    header[16:19] = [1, 2, 3]
    header_f[19:22] = [data.min(), data.max(), data.mean()]
    header[52] = 542130509
    header[53] = 16708
    with open(fname, 'wb') as (f):
        header.tofile(f)
        n.reshape(data, (-1, ), order='F').tofile(f)


def readMRCimgs(fname, idx, num=None):
    hdr = readMRCheader(fname)
    nx = hdr['nx']
    ny = hdr['ny']
    nz = hdr['nz']
    datatype = hdr['datatype']
    assert idx < nz
    if num == None:
        num = nz - idx
    assert idx + num <= nz
    assert num > 0
    datasizes = {1: 2, 2: 4}
    with open(fname) as (f):
        f.seek(1024 + idx * datasizes[datatype] * nx * ny)
        if datatype == 0:
            return n.reshape(n.fromfile(f, dtype='int8', count=nx * ny * num), (nx, ny, num), order='F')
        if datatype == 1:
            return n.reshape(n.fromfile(f, dtype='int16', count=nx * ny * num), (nx, ny, num), order='F')
        if datatype == 2:
            return n.reshape(n.fromfile(f, dtype='float32', count=nx * ny * num), (nx, ny, num), order='F')
        assert False, ('Unsupported MRC datatype: {0}').format(datatype)
    return


def readMRCheader(fname):
    hdr = None
    with open(fname) as (f):
        hdr = {}
        header = n.fromfile(f, dtype=n.int32, count=256)
        header_f = header.view(n.float32)
        hdr['nx'], hdr['ny'], hdr['nz'], hdr['datatype'] = header[:4]
        hdr['xlen'], hdr['ylen'], hdr['zlen'] = header_f[10:13]
    return hdr


def readMRCmemmap(fname, inc_header=False):
    hdr = readMRCheader(fname)
    nx = hdr['nx']
    ny = hdr['ny']
    nz = hdr['nz']
    dtype = {0: n.int8, 1: n.int16, 2: n.float32}[hdr['datatype']]
    mm = n.memmap(fname, dtype, 'r', offset=1024, shape=(nx, ny, nz), order='F')
    if inc_header:
        return (mm, hdr)
    return mm


class LazyMRC:

    def __init__(self, fname, shape, dtype, idx):
        self.fname = fname
        self.shape = (int(shape[0]), int(shape[1]))
        self.idx = idx
        self.dtype = dtype
        self.length = n.dtype(dtype).itemsize * shape[0] * shape[1]
        self.offset = 1024 + idx * self.length

    def get(self):
        with open(self.fname) as (f):
            f.seek(self.offset)
            data = n.reshape(n.fromfile(f, dtype=self.dtype, count=n.prod(self.shape)), self.shape, order='F')
        return data

    def view(self):
        return self.get()


def get_dtype(hdr):
    return {1: n.int16, 2: n.float32}[hdr['datatype']]


def readMRClazy(fname):
    hdr = readMRCheader(fname)
    shape = (hdr['nx'], hdr['ny'])
    num = hdr['nz']
    dtype = get_dtype(hdr)
    lazy_data = [ LazyMRC(fname, shape, dtype, idx) for idx in range(num) ]
    return lazy_data