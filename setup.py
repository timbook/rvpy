from setuptools import setup
import sys

if sys.version_info < (3, 6):
    sys.exit("Sorry, this package requires at least Python 3.6")

setup(name="rvpy",
      version="0.3",
      description="Working with random variables in an OOish way.",
      url="https://github.com/timbook/rvpy",
      author="Tim Book",
      author_email="TimothyKBook@gmail.com",
      license="MIT",
      packages=['rvpy'],
      install_requires=['numpy', 'sklearn'],
      zip_safe=False)
