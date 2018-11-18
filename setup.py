from setuptools import setup

setup(
    name='pyebsd',
    version='0.1dev',
    description='',
    author='Arthur Nishikawa',
    author_email='nishikawa.poli@gmail.com',
    url='https://github.com/arthursn/pyebsd',
    packages=['pyebsd', 'pyebsd.ebsd', 'pyebsd.crystal', 
              'pyebsd.io', 'pyebsd.selection', 'pyebsd.draw',
              'pyebsd.misc'],
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'pandas',
                      'scipy', 'pillow'],
    long_description=open('README.md').read(),
)
