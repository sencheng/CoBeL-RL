# basic imports
from setuptools import setup, find_packages
from pkg_resources import DistributionNotFound, get_distribution


# function for checking installed packages
def get_dist(pkgname):
    try:
        return get_distribution(pkgname)
    except DistributionNotFound:
        return None

with open('readme.md', 'r') as fh:
    long_description = fh.read()
    
# define requirements
requirements = [
        'gymnasium>=0.28.0',
        'shapely>=2.0.0',
        'scikit-learn>=1.2.2',
        'testresources',
        'pyqtgraph>=0.12.0',
        'PyQt5>=5.15.0',
        'scipy'
    ]

# define extra requirements (i.e. Tensorflow, Torch, etc.)
requirements_extra = {
        'keras-rl': ['tensorflow>=2.12.0', 'keras-rl2'],
        'tensorflow': ['tensorflow>=2.12.0'],
        'torch': ['torch>=2.0.0'],
        'unity': ['mlagents>=0.24.0', 'mlagents-envs>=0.24.0'],
        }

# ensure that only one opencv-python version is installed (opencv-python-headless per default)
opencv_dist = get_dist('opencv-python')
opencv_headless_dist = get_dist('opencv-python-headless')
if opencv_dist is None and opencv_headless_dist is None:
    requirements.append('opencv-python-headless>=4.7.0.0')
elif opencv_dist is not None:
    requirements.append('opencv-python>=4.7.0.0')
else:
    requirements.append('opencv-python-headless>=4.7.0.0')

# setup package
setup(
    name='cobel',
    version='2.1.0',
    keywords='reinforcement learning, neuroscience, simulation',
    url='https://github.com/sencheng/CoBeL-RL',
    description='Closed-loop simulator of complex behavior and learning based on reinforcement learning and deep neural networks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(include=['cobel', 'cobel.*']),
    install_requires=requirements,
    extras_require=requirements_extra,
    python_requires='>=3.10',
    author='Sen Cheng',
    author_email="sen.cheng@rub.de",
    classifiers=['Programming Language :: Python :: 3.10',
                 'Programming Language :: Python :: 3.11',
                 'Operating System :: OS Independent']
)
