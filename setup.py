# basic imports
from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution


# function for checking installed packages
def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

with open("readme.md", "r") as fh:
    long_description = fh.read()
    
# define requirements
requirements = [
        'gym>=0.9.2',
        'shapely>=1.7.0',
        'testresources',
        'pyqtgraph>=0.10.0',
        'PyQt5>=5.12.3',
        'PyQt5-sip>=12.7.1',
        'scipy'
    ]

# define extra requirements (i.e. Tensorflow, Torch, etc.)
requirements_extra = {
        'keras-rl': ['tensorflow>=2.1.0', 'keras-rl2'],
        'tensorflow': ['tensorflow>=2.1.0'],
        'torch': ['torch>=1.12.1'],
        'unity': ['mlagents>=0.24.0', 'mlagents-envs>=0.24.0'],
        }

# ensure that only one opencv-python version is installed (opencv-python-headless per default)
opencv_dist = get_dist('opencv-python')
opencv_headless_dist = get_dist('opencv-python-headless')
if opencv_dist is None and opencv_headless_dist is None:
    requirements.append('opencv-python-headless>=4.2.0.32')
elif opencv_dist is not None:
    requirements.append('opencv-python>=4.2.0.32')
else:
    requirements.append('opencv-python-headless>=4.2.0.32')

# setup package
setup(
    name='cobel',
    version='2.0.0',
    keywords='reinforcement learning, neuroscience, simulation',
    url='https://github.com/sencheng/CoBeL-RL',
    description='Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['cobel', 'cobel.*']),
    install_requires=requirements,
    extras_require=requirements_extra,
    python_requires=">=3.6",
    author="Sen Cheng",
    author_email="sen.cheng@rub.de",
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"]
)
