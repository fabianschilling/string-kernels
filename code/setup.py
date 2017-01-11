from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=[
        Extension("ssk", ["ssk.c"],
                  include_dirs=[numpy.get_include()]),
    ],
)

# setup(
    # ext_modules=cythonize('ssk.pyx'),
    # include_dirs=[numpy.get_include()]
# )
