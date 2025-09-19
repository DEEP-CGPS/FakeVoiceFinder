from setuptools import setup, find_packages

setup(
    name="FakeVoiceFinder",
    version="0.1.0",
    url="https://github.com/DEEP-CGPS/FakeVoiceFinder",
    license="MIT",
    packages=find_packages(exclude=["notebooks", "tests", "outputs"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn",
        "librosa",
        "pywavelets",
        "pillow",
        "torch>=2.5,<3.0",
        "torchvision>=0.20,<1.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
