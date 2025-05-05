from setuptools import setup, find_packages

setup(
    name="sundas",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    author="Your Name",
    description="A simple package containing fil.py",
    long_description=open("README.md").read() if __name__ == "__main__" else "",
    long_description_content_type="text/markdown",
    url="https://github.com/commitEarth/Sundas",
)
