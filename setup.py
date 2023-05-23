import setuptools
from distutils.core import Extension
import pathlib
import codecs

HERE = pathlib.Path(__file__).resolve().parents[0]


def get_version(path):
    with codecs.open(path, 'r') as fp:
        for line in fp.read().splitlines():
            if line.startswith('__version__'):
                delim = '"' if '"' in line else "'"
                return line.split(delim)[1]
        else:
            raise RuntimeError("Unable to find version string.")


clib_beam = Extension(
    name='pyant.clibbeam',
    sources=[
        'src/clibbeam/array.c',
        'src/clibbeam/beam.c',
    ],
    include_dirs=['src/clibbeam/'],
)


setuptools.setup(
    version=get_version(HERE / 'src' / 'pyant' / 'version.py'),
    ext_modules=[clib_beam],
)
