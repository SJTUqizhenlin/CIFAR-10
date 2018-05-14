from distutils.core import setup 
from Cython.Build import cythonize 

setup(
    name = "convolute functions",
    ext_modules = cythonize("conv_func.pyx")
)