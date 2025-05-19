from setuptools import setup, find_packages

setup(
    name="image-lowpass-filters",
    version="0.1.2",
    author="Cassini Wei",
    author_email="cassiniwei@outlook.com",
    description="Ideal, Butterworth and Gaussian frequency‐domain filters for PyTorch",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/CassiniHuy/image-low-pass-filters-pytorch",
    license="MIT",
    license_files=("LICENSE",),
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.7.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
