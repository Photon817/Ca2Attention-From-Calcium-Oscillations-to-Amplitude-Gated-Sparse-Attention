"""CORC: Calcium-inspired Oscillatory Reservoir Computing framework."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="corc",
    version="2.0.0",
    author="CORC Authors",
    description="Calcium-inspired Oscillatory Reservoir Computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/corc",
    packages=find_packages(include=["corc_v2", "corc_v2.*", "agsc_proof", "agsc_proof.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "agsc": ["torch>=1.10.0"],
        "all": ["torch>=1.10.0"],
    },
    keywords="reservoir-computing, calcium-oscillations, attention, biological-computation, neuroscience",
)