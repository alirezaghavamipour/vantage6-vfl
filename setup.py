from setuptools import setup, find_packages

setup(
    name="vantage6-vfl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["vantage6-algorithm-tools"],
)
