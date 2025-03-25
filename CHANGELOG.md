# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Tests
- Definition and interface for HDFImage format
- New interfaces based on the pure-NumPy core logics

### Changed

- Replace the C-dependent modules with a pure NumPy modules
- Move the old `geometry` class to a separate module
- Drop dependency on SciPy

### Removed

- All Cython codes
- setup.py
- refine.py

## [1.1.1] - 2023-03-05
