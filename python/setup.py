"""
FusionML - High-Performance ML Framework for Apple Silicon
"""

from setuptools import setup, find_packages

setup(
    name="fusionml",
    version="0.1.0",
    description="High-Performance ML Framework for Apple Silicon with GPU+CPU parallel execution",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="FusionML Team",
    author_email="fusionml@example.com",
    url="https://github.com/yourname/fusionml",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "metal": ["pyobjc-framework-Metal>=9.0"],
        "dev": ["pytest", "black", "mypy"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: MacOS :: MacOS X",
    ],
    keywords="machine-learning deep-learning apple-silicon gpu neural-network",
)
