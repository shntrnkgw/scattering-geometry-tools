# coding=utf-8
"""I/O interface for a generic multilayer array collection

The code was initially in `snpl` package and has been included
in this package for convenience. 

Note:
    As of version 1.0.0, the use of NpzImage format to save the file is discouraged, 
    as it seems to be causing problems (and it is can be unsafe also). 
    Use HDF5Image, which is actually just a HDF5 file. 
"""

import io
import numpy as np

class NpzImage:
    '''I/O interface for NpzImage file. 
    
    NpzImage is a convenient file format to store multi-layered multi-dimensional arrays with a metadata header. 
    Multiple ``numpy.ndarray`` objects can be stored as "layers", which can be specified by a "key". 

    Args:
        fp (str or file-like): Path or file-like object of the source file. If None, an empty object is created. 

    Note:
        - Shape restriction on the layer items has been removed. Layers with different shapes can now be mixed. 
        - Layer keys must be a string. 
        - "h" cannot be used as a layer key. 

    Examples:
        Creation

        >>> im = snpl.image.NpzImage()
        
        Adding headers
        
        >>> im.h["number"] = 100.0
        >>> im.h["string"] = "wow",
        >>> im.h["bool"] = True
        >>> im.h["list"] = [1.0, 2.0]

        Adding layers

        >>> im.layers["one"] = np.array( [[1.0, 2.0], [3.0, 4.0]] )
        >>> im.layers["two"] = np.array( [[5.0, 6.0], [7.0, 8.0]] )
        
        Save to a file

        >>> im.save("npzimage.npz")

        Load from a file

        >>> im2 = snpl.image.NpzImage("npzimage.npz")
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
        
        self.h: dict = {"history": []}
        self.layers: dict[str,np.ndarray] = {}
        
        if fp:
            self.fromfile(fp)


    def fromfile(self, fp: str|io.FileIO|io.BytesIO) -> None:
        with np.load(fp, allow_pickle=True) as z:
            for key, arr in z.items():
                if key == "h":
                    self.h = {k: v for k, v in arr[()].items()}
                else:
                    self.layers[key] = arr

    def get_layer(self, key: str) -> np.ndarray:
        return self.layers[key]

    def append_layer(self, key: str, arr: np.ndarray) -> None:
        self.layers[key] = arr
    
    def pop_layer(self, key: str) -> np.ndarray:
        return self.layers.pop(key)
    
    def save(self, fp: str|io.FileIO|io.BytesIO, compress: bool=False) -> None:
        h: dict = {k: v for k, v in self.h.items()}
        if compress:
            np.savez_compressed(fp, h=h, **self.layers)
        else:
            np.savez(fp, h=h, **self.layers)

    def append_history(self, string: str) -> None:
        try:
            self.h["history"].append(string)
        except KeyError:
            self.h["history"] = [string, ] 