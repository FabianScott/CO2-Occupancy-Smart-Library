from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1",
    description="tools for occupancy estimation",
    author="Smart Library",
    packages=["utils"],  # same as name
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
    ],  # external packages as dependencies
)
