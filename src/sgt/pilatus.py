# coding=utf-8
'''
Created on 2017/11/17

@author: snakagawa
'''

import struct
import numpy as np
from typing import Tuple, IO

def _splitnull(c_string: str) -> str:
    """
    Converts null terminated string to a string. 
    """
    buf = c_string.split("\0")
    return buf[0]

DTYPES: dict = {1: "B", 
                2: "s", 
                3: "H", 
                4: "L", 
                6: "b", 
                7: "c", 
                8: "h", 
                9: "l", 
                11: "f", 
                12: "d"}

def ReadTiff(fp: str|IO) -> np.ndarray:
    """Reads a .tif file from PILATUS detector and returns the 2-d data as a numpy array. 

    Args:
        fp: file-like or path to the file. 
    
    Returns:
        A 2D array of the int type. 

    Example:
        >>> arr = pilatus.ReadTiff("AgBh.tif")
        >>> print(arr.shape, arr.dtype)
    """
    f: IO|None = None
    close: bool = True
    byteorder: str = ""
    version: int = 0
    ifd_offset: int = 0
    entry_count: int = 0
    tag_dict: dict = {}
    entry_number: int = 0
    
    tag_code: int = 0
    dtype_code: int = 0
    count: int = 0
    data_str: str = ""
    data_offset: int = 0
    c_offset: int = 0

    dtype: str = ""
    numer: int = 0
    denom: int = 0
    data: float = 0.0

    dformat: str = ""
    dlength: int = 0


    ImageWidth: int = 0
    ImageHeight: int = 0
    BitsPerSample: Tuple[int, ...] = ()
    Compression: int = 0
    PhotometricInterpretation: int = 0
    StripOffsets: Tuple[int, ...]
    RowsPerStrip: int = 0
    StripBytesCounts: Tuple[int, ...] = ()
    XResolution: float = 0.0
    YResolution: float = 0.0
    NewSubfileType: int = 0
    DateTime: bytes = b""

    BitsPerSampleSingle: int = 0
    ImageDataOffset: int = 0
    ImageBytesCount: int = 0

    #------#
    # Open #
    #------#
    if isinstance(fp, str):
        f = open(fp, "rb")
        close = True
    else:
        f = fp
        close = False
    
    # f.seek(0)
    
    #--------#
    # Header #
    #--------#
    try:
        byteorder = {b'II': '<', b'MM': '>'}[f.read(2)]
    except KeyError:
        raise IOError("not a valid TIFF file")
    
    version, = struct.unpack(byteorder+"H", f.read(2))
    if version != 42:
        raise IOError("not a valid TIFF file")
    
    ifd_offset, = struct.unpack(byteorder+"l", f.read(4))
    
    #-----#
    # IFD #
    #-----#
    
    ### SEEK
    f.seek(ifd_offset)
    # number of entries
    entry_count, = struct.unpack(byteorder+"H", f.read(2))
    tag_dict = {}
    for entry_number in range(entry_count):
        
        tag_code, dtype_code, count, data_str = struct.unpack(byteorder+"HHl4s", f.read(12))
        # special data types (RATIONAL and SRATIONAL)
        if dtype_code in (5, 10) and count == 1:
            data_offset, = struct.unpack(byteorder+"L", data_str)
            ### POP
            c_offset = f.tell()
            f.seek(data_offset)
            if dtype_code == 5:
                dtype = "R"
                numer, denom = struct.unpack(byteorder+"LL", f.read(8))
            else:
                dtype = "S"
                numer, denom = struct.unpack(byteorder+"lL", f.read(8))
            data = float(numer)/denom    
            f.seek(c_offset)
            ### PUSH
        else:
            try:
                dtype = DTYPES[dtype_code]
            except KeyError:
                raise IOError("Contains unknown or unsupported data type")
            
            dformat = "{0}{1}{2}".format(byteorder, count, dtype)
            dlength = struct.calcsize(dformat)
            if dlength <= 4:
                data = struct.unpack(dformat, data_str[:dlength])
                data_offset = 0
            else:
                data_offset, = struct.unpack(byteorder+"L", data_str)
                ### POP
                c_offset = f.tell()
                f.seek(data_offset)
                data = struct.unpack(dformat, f.read(dlength))
                f.seek(c_offset)
                ### PUSH
                if type(data[0]) == str:
                    data = _splitnull(data[0])
        
        #print tag_code, dtype, count, data
        
        tag_dict[tag_code] = data
        
    #------------#
    # Check tags #
    #------------#
    try:
        ImageWidth    = tag_dict[256][0]
        ImageHeight   = tag_dict[257][0]
        BitsPerSample = tag_dict[258]
        Compression   = tag_dict[259][0]
        PhotometricInterpretation = tag_dict[262][0]
        StripOffsets  = tag_dict[273]
        RowsPerStrip  = tag_dict[278][0]
        StripBytesCounts = tag_dict[279]
        XResolution   = tag_dict[282]
        YResolution   = tag_dict[283]
        
        NewSubfileType = tag_dict[254][0]
        #ImageDescription = tag_dict[270][0]
        #Model         = tag_dict[272][0]
        #Software      = tag_dict[305]
        DateTime      = tag_dict[306][0]
        #Artist        = tag_dict[315]
        
        # print("ImageWidth", ImageWidth, type(ImageWidth))
        # print("ImageHeight", ImageHeight, type(ImageHeight))
        # print("BitsPerSample", BitsPerSample, type(BitsPerSample))
        # print("Compression", Compression, type(Compression))
        # print("PhotometricInterpretation", PhotometricInterpretation, type(PhotometricInterpretation))
        # print("StripOffsets", StripOffsets, type(StripOffsets))
        # print("RowsPerStrip", RowsPerStrip, type(RowsPerStrip))
        # print("StripBytesCounts", StripBytesCounts, type(StripBytesCounts))
        # print("XResolution", XResolution, type(XResolution))
        # print("YResolution", YResolution, type(YResolution))
        # print("NewSubfileType", NewSubfileType, type(NewSubfileType))
        # print("DateTime", DateTime, type(DateTime))

    except KeyError:
        raise IOError("Missing tag(s)")
    
    if len(BitsPerSample) > 1:
        raise IOError("Unsupported tiff (color): BitsPerSample=" + ", ".join([str(e) for e in BitsPerSample]))
    else:
        BitsPerSampleSingle = BitsPerSample[0]
    
    if Compression != 1:
        raise IOError("Unsupported tiff (compressed)")
    
    if PhotometricInterpretation != 1:
        raise IOError("Unsupported tiff (wrong photometric interpretation)")
    
    if len(StripOffsets) != 1:
        raise IOError("Unsupported tiff (multiple strips)")
    else:
        ImageDataOffset = StripOffsets[0]
    
    ImageBytesCount = StripBytesCounts[0]
    
    if NewSubfileType != 0:
        raise IOError("Unsupported tiff (multipage or mask)")
        
    #-----------#
    # Load data #
    #-----------#
    f.seek(ImageDataOffset)
    PixelNumber = ImageWidth * ImageHeight
    if BitsPerSampleSingle == 32:
        flat = np.fromfile(f, dtype=np.int32, count=PixelNumber)
        image = np.reshape(flat, (ImageHeight, ImageWidth), order="C")
    
    if close:
        f.close()
    
    return image

# def LoadPilatusTiff(fp, key_dest="intensity"):
#     im = ReadGrayTiff(fp)
#     wim = snpl.image.NpzImage()
#     wim.append_layer(key_dest, im)
#     return wim
# 
# def get_mask_NpzImage(fp, empty=False):
#     im = ReadGrayTiff(fp)
#     wim = snpl.image.NpzImage()
#     mask = np.zeros_like(im, dtype=np.uint8)
#     if not empty:
#         mask[np.where(im<0)] = 1
#     wim.append_layer("mask", mask)
#     return wim

if __name__ == '__main__':
    im = ReadTiff("../test/scatter/AgBh.tif")