#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="elasticros",
    version="1.0.0",
    author="ElasticROS Team",
    author_email="elasticros@example.com",
    description="Algorithm-level elastic computing framework for IoRT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ElasticROS/elasticros",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Robotics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "boto3>=1.17.0",
        "pyyaml>=5.3.0",
        "psutil>=5.7.0",
        "matplotlib>=3.3.0",
        "pandas>=1.1.0",
        "paramiko>=2.7.0",
        "cryptography>=3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=20.8b1",
            "flake8>=3.8",
        ],
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "elasticros-setup=scripts.setup_aws:main",
            "elasticros-monitor=elasticros_core.utils.metrics:monitor_main",
        ],
    },
    include_package_data=True,
    package_data={
        "elasticros": ["config/*.yaml", "cloud_integration/templates/*.yaml"],
    },
)