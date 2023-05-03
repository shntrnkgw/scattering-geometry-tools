# coding="utf-8"

import sgt.npzimage
import numpy as np

if __name__ == "__main__":
    im = sgt.npzimage.NpzImage()
    im.h["number"] = 100.0
    im.h["string"] = "wow"
    im.h["bool"] = True
    im.h["list"] = [1.0, 2.0]
    im.append_layer("one", np.array( [[1.0, 2.0], [3.0, 4.0]] ) )
    im.append_layer("two", np.array( [[5.0, 6.0], [7.0, 8.0]] ) )
    
    im.save("npzimage.npz")
    
    im2 = sgt.npzimage.NpzImage("npzimage.npz")
    print(im2.h)
    print(im2.layers["one"])
    print(im2.layers["two"])
    print(im2.h["string"])