#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="Copamundial",
    version="1.0.0",
    description="Copamundial - protein function prediction algorithm based on coembedding of PPI networks of multiple species into a single vector space.",
    author="Kapil Devkota, Grigoriy Sterin",
    author_email="kapil.devkota@tufts.edu, gr.sterin@gmail.com",
    license="MIT",
    packages=find_packages(where="."),
    entry_points={
        "console_scripts": [
            "copamundial = copamundial.copamundial_main:main",
        ],
    },
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "biopython",
        "matplotlib",
        "seaborn",
        "tqdm",
        "scikit-learn",
        "networkx",
        "glidetools",
        "torch",
        "pyyaml"
    ]
)