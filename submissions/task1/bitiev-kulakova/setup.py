from setuptools import setup, find_packages

PACKAGE = "matrixgame"
NAME = "Matrix_Game"
DESCRIPTION = "Solution of the matrix game by the simplex method"
AUTHOR = "A.Bitiev, M.Kulakova"
VERSION = __import__(PACKAGE).__version__
 
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    packages=find_packages(exclude=["tests.*", "tests"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Framework :: Django",
    ],
)
