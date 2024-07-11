# kspec-gfa

![Versions](https://img.shields.io/badge/python-3.9+-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/kspec-gfa/badge/?version=latest)](https://kspec-gfa.readthedocs.io/en/latest/?badge=latest)

# The KSPEC GFA Camera Controller
- KSPEC-GFA is a tool for guiding, focusing, and acquisition sequence control in KSPEC observation.
- The Controller communicate with Basler Guide cameras for guiding and focusing processes.
- The Controller use the [pypylon](https://github.com/basler/pypylon) library as the middleware for the communication.

# Getting Started

## Installation

`kspec-gfa` can be installed using by cloning this repository

```console
git clone https://mmingyeong@bitbucket.org/mmmingyeong/kspec-gfa.git
```

The preferred installation for development is using [poetry](https://python-poetry.org/)

```console
cd kspec-gfa
poetry install
```

## Quick Start

```console
cd python/kspec-gfa/commander
python status.py
```

If you want to know the usage of each command, use --help option.

```console
python grab.py --help
```

## Software Architecture

Here is the Software Architecture diagram explaining the hierarchy of KSPEC-GFA.

![img](./docs/sphinx/source/_static/kspec-gfa_software_architecture.png)
