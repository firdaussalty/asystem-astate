from setuptools import setup, find_packages
import os

setup(
    name="astate",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'astate._core': ['*.so'],
    },
    install_requires=[
        'torch>=1.0.0',
        'numpy>=1.19.0',
    ],
    author="Astate Team",
    description="Python interface for Astate library",
    python_requires='>=3.9',
)