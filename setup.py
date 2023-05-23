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


libbeam = Extension(
    name='pyant.libbeam',
    sources=[
        'src/libbeam/libarray.c',
        'src/libbeam/libbeam.c',
    ],
    include_dirs=['src/libbeam/'],
)


setuptools.setup(
    version=get_version(HERE / 'src' / 'pyant' / 'version.py'),
    ext_modules=[libbeam],
)
