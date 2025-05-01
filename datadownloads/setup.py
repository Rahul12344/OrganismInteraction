from setuptools import setup, find_packages

setup(
    name="datadownloads",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "requests",
        "beautifulsoup4",
        "pandas",
        # Add other dependencies from your requirements.txt
    ],
)