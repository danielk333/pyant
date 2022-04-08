import setuptools
from distutils.core import Extension

libbeam = Extension(
    name='pyant.libbeam',
    sources=['src/libbeam/libarray.c'],
    include_dirs=['src/libbeam/'],
)

setuptools.setup(
    package_dir={
        "": "src"
    },
    packages=setuptools.find_packages(where="src"),
    ext_modules=[libbeam],
)
