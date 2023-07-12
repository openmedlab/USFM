#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="USFM",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="GeorgeJiao",
    author_email="georgejiao8@gmail.com",
    url="https://github.com/openmedlab/USFM",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(exclude=("configs", "tools", "demo", "tests")),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = usfm.train:main",
            "eval_command = usfm.eval:main",
        ]
    },
)
