# coding="utf-8"
"""I/O interface for a generic multilayer array collection using the HDF5 format
"""

import h5py
import numpy as np
import io

class HDF5Image:
    '''I/O interface for the scattering image data in HDF5 format. 
    
    This class stores multiple multi-dimensional arrays with a metadata header, 
    which can be read from or written to an HDF5 file. 
    Multiple ``numpy.ndarray`` objects can be stored as "layers", which can be specified by a "key". 
    The metadata is stored as a dictionary. 
    The scattering images in scattering-geometry-tools have been previously stored in NpzImage format
    which heavily relied on numpy's `save` function. 
    However, the long-term readability of this kind of file was not guaranteed. 
    So the HDF5 format, which was more common, reliable, and very likely to be maintained for a long time, 
    was newly adopted. 

    Args:
        fp (str or file-like): Path or file-like object of the source file. If None, an empty object is created. 

    Note:
        - There is no shape restriction on the layer items. Layers with different shapes can be mixed. 
        - Currently, we do not use group functionality in HDF5. 
        The layers are stored as datasets in the root group. 
        The metadata are stored in attr of the root group. 

    Examples:
        Creation

        >>> im = hdf5image.HDF5Image()
        
        Adding headers
        
        >>> im.h["number"] = 100.0
        >>> im.h["string"] = "wow",
        >>> im.h["bool"] = True
        >>> im.h["list"] = [1.0, 2.0]

        Adding layers

        >>> im.layers["one"] = np.array( [[1.0, 2.0], [3.0, 4.0]] )
        >>> im.layers["two"] = np.array( [[5.0, 6.0], [7.0, 8.0]] )
        
        Save to a file

        >>> im.save("hdf5image.hdf5")

        Load from a file

        >>> im2 = hdf5image.HDF5Image("hdf5image.hdf5")
        >>> print(im2.h)
        {'number': 100.0, 'string': 'wow', 'bool': True, 'list': [1.0, 2.0], 'version': '0.3.0'}
        >>> print(im2.layers["one"])
        [[1. 2.]
        [3. 4.]]
        >>> print(im2.layers["two"])
        [[5. 6.]
        [7. 8.]]
        >>> print(im2.h["string"])
        wow
    '''

    def __init__(self, fp: str|io.FileIO|io.BytesIO|None=None) -> None:
        
        self.h: dict = {}
        self.layers: dict[str,np.ndarray] = {}
        
        if fp:
            self.fromfile(fp)

    def fromfile(self, fp: str|io.FileIO|io.BytesIO) -> None:
        with h5py.File(fp) as h5f:
            self.h = {k: v for k, v in h5f.attrs.items()}
            for key in h5f.keys():
                self.layers[key] = h5f[key][:] # get as numpy array

    def get_layer(self, key: str) -> np.ndarray:
        return self.layers[key]

    def append_layer(self, key: str, arr: np.ndarray) -> None:
        self.layers[key] = arr
    
    def pop_layer(self, key: str) -> np.ndarray:
        return self.layers.pop(key)
    
    def save(self, fp: str|io.FileIO|io.BytesIO, compress: bool=True) -> None:
        
        kwargs: dict = {}
        if compress:
            kwargs["compression"] = "gzip"

        with h5py.File(fp, mode="w") as wf:
            wf.attrs.update(self.h)
            for name, data in self.layers.items():
                wf.create_dataset(name=name, data=data, **kwargs)

    def append_history(self, string: str) -> None:
        try:
            self.h["history"].append(string)
        except KeyError:
            self.h["history"] = [string, ]

if __name__ == "__main__":
    pass