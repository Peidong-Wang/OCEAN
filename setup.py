from setuptools import find_packages, setup
from ocean import __version__

setup(
    name='ocean',
    version=__version__,
    packages=find_packages(exclude=['egs']),
    install_requires=[
        'numpy>=1.14',
        'torch>=1.7',
        'torchvision>=0.4',
    ],
    url='https://github.com/Peidong-Wang/OCEAN',
    license='Apache-2.0',
)
