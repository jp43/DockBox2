import sys
from setuptools import setup

setup(name='dockbox2',
    version='1.0',
    packages=['dockbox2'],
    scripts=['bin/rundbx2', 'bin/split_train_val_dbx2'],
    license='LICENSE',
    description='GNN docking-based binding mode and affinity predictor',
    long_description=open('README.md').read(),
)
