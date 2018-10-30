from setuptools import setup

setup(name="rvpy",
      version="0.1",
      description="Working with random variables in an OOish way.",
      url="https://github.com/timbook/rvpy",
      author="Tim Book",
      author_email="TimothyKBook@gmail.com",
      license="MIT",
      packages=['rvpy'],
      install_requires=['numpy', 'sklearn'],
      zip_safe=False)
