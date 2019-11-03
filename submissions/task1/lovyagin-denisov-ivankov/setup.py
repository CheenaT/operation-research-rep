from setuptools import setup, find_packages
import matrgame

setup(name='nashequilibrium',
      version=matrgame.__version__,
      description='Find optimal strategies for players A and B',
      url='https://github.com/r4ndompuff/Tupo_prak',
      author='Pozhilaya salamandra',
      author_email='nick_denisov_777@mail.ru',
      license='MSU',
      packages=find_packages(),
      keywords=['saddle point','matrix game'],
      install_requires=['NumPy','matplotlib']
)
