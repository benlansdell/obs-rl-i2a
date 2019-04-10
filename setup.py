from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "This repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("o2a", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='o2a',
    py_modules=['o2a'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle==0.5.2',
        'gym[atari,box2d,classic_control]>=0.10.8',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn==0.8.1',
        'tensorflow-gpu==1.12.0',
        'tqdm'
    ],
    description="Observation to action transfer in RL.",
    author="Ben Lansdell",
)
