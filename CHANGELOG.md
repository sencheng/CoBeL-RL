# Changelog

## 3.0.1

Released on 2026-03-20.

### Bug fixes

 - Fixed various type hinting errors
 - Fixed a bug in the SFMAMemory class' `retrieve_random_batch` function
 - Fixed network projections not being flattened in the MFEC class
 - Fixed the DQN class not tracking the number of steps since the last network update
 - Fixed an assertion error that would occur when using the DynaDQN class' DDQN mode
 - Fixed a crash when using the DynaDSR class due to Q-values not being retrieved properly
 - Fixed multimodal observations not being properly stored in the ADQN class
 - Fixed a bug in the PMA agent due to the use of a deprecated NumPy feature
 - Fixed selected actions not being logged correctly in the Sequence class' `step` function
 - Fixed the TorchNetwork and FlexibleTorchNetwork classes' `get_weights` function returning a reference to instead of a copy of the network weights

### Documentation

 - Fixed documentation errors throughout the framework
 - Moved to using the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) convention for docstrings

### Other changes

 - Added unit tests for various parts of the framework
 - Added pydocstyle rules to Ruff configuration
 - Moved to version `0.10.x` of `uv_build`
 - Removed deprecated license classifier ([PEP 639](https://peps.python.org/pep-0639/))
 - Refactored demo scripts
 - Added change log

### Contributors

 - [Nicolas Diekmann](https://github.com/nicolasdiekmann)

## 3.0.0

Released on 2025-10-20.

Major refactoring of the project with many breaking changes.

### Framework changes

 - Switch from using a `setup.py` script to `pyproject.toml` for packaging ([PEP 621](https://peps.python.org/pep-0621/))
 - Switch build system from [setuptools](https://setuptools.pypa.io/en/latest/) to [uv_build](https://docs.astral.sh/uv/concepts/build-backend/)
 - Removed many convoluted inter-class dependencies
 - Simplified framework structure
 - Introduced consistent type hints
 - Added [Ruff](https://docs.astral.sh/ruff/) for linting and formatting
 - Extended documentation
 - Removed unsupported and dead packages, e.g., `keras-rl2`

### Contributors

 - [Nicolas Diekmann](https://github.com/nicolasdiekmann)
 - [Sandhiya Vijayabaskaran](https://github.com/sandhiyavb)
 - [Jon Recalde](https://github.com/jonrekalde)
 - Gerrit Simon Fischer
 - Duc Cuong Tommy Tran
 - Maximilian Wojak
 - Sebastian Benedict Schäfer
 - David Jarne Nörtemann
 - Oleksandr Chaban

