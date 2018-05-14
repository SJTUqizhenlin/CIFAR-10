from distutils.core import setup 
from Cython.Build import cythonize 

setup(
    name = "maxpool functions",
    ext_modules = cythonize("pool_func.pyx")
)