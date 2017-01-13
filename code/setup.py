# setup.py

from distutils.core import setup, Extension

ssk_module = Extension('_ssk', sources=['ssk.c', 'ssk.i'])

setup(name='ssk', ext_modules=[ssk_module], py_modules=["ssk"])
