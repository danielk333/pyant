import os
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install
import subprocess

with open('README.rst', 'r') as fh:
    long_description = fh.read()


with open('requirements', 'r') as fh:
    pip_req = fh.read().split('\n')
    pip_req = [x.strip() for x in pip_req if len(x.strip()) > 0]


setuptools.setup(
    name='pyant',
    version='0.1.0',
    long_description=long_description,
    url='https://github.com/danielk333/pyant',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT',
        'Operating System :: OS Independent',
    ],
    install_requires=pip_req,
    packages=setuptools.find_packages(),
    extras_require={'plotting': ['matplotlib>=3.2.0']},
    # metadata to display on PyPI
    author='Daniel Kastinen',
    author_email='daniel.kastinen@irf.se',
    description='pyant',
    license='MIT',
)
