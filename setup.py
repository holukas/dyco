from distutils.core import setup

setup(
    name='dyla',
    packages=['dyla'],
    version='0.0.1',
    license='GNU General Public License v3.0',
    description='A Python package to detect and compensate for shifting lag times in ecosystem time series',
    author='Lukas Hörtnagl',
    author_email='lukas.hoertnagl@usys.ethz.ch',
    url='https://gitlab.ethz.ch/holukas/dyla-dynamic-lag-remover',
    download_url='XXX',  # todo
    keywords=['ecosystem', 'eddy covariance', 'fluxes',
              'time series', 'lag', 'timeshift'],
    install_requires=['pandas', 'numpy', 'matplotlib', ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3.0',
        'Programming Language :: Python :: 3.6',
    ],
)
