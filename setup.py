from setuptools import setup

setup(
    name='pyebsd',
    version='0.1dev',
    description='python library for post-processing of Electron Backscattered Diffraction (EBSD) data',
    author='Arthur Nishikawa',
    author_email='nishikawa.poli@gmail.com',
    url='https://github.com/arthursn/pyebsd',
    packages=['pyebsd', 'pyebsd.ebsd', 'pyebsd.io',
              'pyebsd.selection', 'pyebsd.draw', 'pyebsd.misc'],
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'pandas',
                      'scipy', 'pillow'],
    long_description=open('README.md').read()
)
