# scattering-geometry-tools
Tools to process scattering data. 

## Inastallation
This package is not (and will not be) distributed on package repositories such as 
PyPI and conda-forge. Please install from the source files on GitHub. 

    $ pip install git+https://github.com/shntrnkgw/scattering-geometry-tools

Or if you do not have Git installed, 

    $ pip install https://github.com/shntrnkgw/scattering-geometry-tools/archive/refs/heads/main.zip

This package contains C extension. 
On Windows, you may have to install C compiler to build and install the extension. 
Get [Microsoft C++ Build Tools] (https://visualstudio.microsoft.com/visual-cpp-build-tools/). 

## Batch build & install for developers
In the root directory, run `devinstall.sh`.
This will build the `.pyx` files into `.c`, 
build `.c`, and install the package in the editable mode. 