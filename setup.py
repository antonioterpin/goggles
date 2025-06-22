"""Setup script for the goggles package."""

import pathlib
import setuptools

setuptools.setup(
    name="goggles",
    version="0.1.0",
    author="Antonio Terpin",
    author_email="aterpin@ethz.ch",
    description="Tools for logging and debugging your robotics research pipeline.",
    url="http://github.com/antonioterpin/goggles",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=["goggles"],
    include_package_data=True,
    install_requires=[
        "wandb[media]",
    ],
    extras_require={
        "dev": ["pytest"],
        "examples": ["matplotlib", "numpy<2"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
