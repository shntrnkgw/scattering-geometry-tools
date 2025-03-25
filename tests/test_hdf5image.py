# coding="utf-8"

import pytest
from sgt import hdf5image
import numpy as np
import os

def test_HDF5Image_create():
    path_temp = "temp.hdf5"

    im = hdf5image.HDF5Image()

    arr1 = np.random.random_sample((400, 300)) - 0.5
    arr2 = np.random.random_sample((200, 300)) - 0.5
    arr3 = np.random.random_sample((100, 300)) - 0.5

    im.layers["arr1"] = arr1
    im.append_layer("arr2", arr2) # now equiv. to layers["arr2"] = arr2
    im.layers["arr3"] = arr3
    
    assert np.all(im.pop_layer("arr3") == arr3)

    im.h["number"] = 100.0
    im.h["文字列キー"] = "文字列値"
    im.h["bool"] = True
    im.h["list"] = [1, 2, 3, 4]

    # save test
    im.save(path_temp, compress=False)

    # load test
    im2 = hdf5image.HDF5Image(path_temp)

    # fromfile test
    im3 = hdf5image.HDF5Image()
    im3.fromfile(path_temp)

    # another fromfile test
    im4 = hdf5image.HDF5Image()
    with open(path_temp, "rb") as fp:
        im4.fromfile(fp)

    os.remove(path_temp)

    # array in file match the input
    for imn in (im2, im3, im4):
        assert np.all(imn.get_layer("arr1") == arr1)
        assert np.all(imn.get_layer("arr2") == arr2)
        assert imn.h["number"] == 100.0
        assert imn.h["文字列キー"] == "文字列値"
        assert imn.h["bool"] == True
        assert imn.h["list"].tolist() == [1, 2, 3, 4]
    
        # raise KeyError for non-existing layer key
        with pytest.raises(KeyError):
            imn.get_layer("this layer key does not exist")
    

if __name__ == "__main__":
    pass