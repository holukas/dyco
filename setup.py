"""
    DYCO Dynamic Lag Compensation
    Copyright (C) 2020  holukas

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='dyco',
    packages=setuptools.find_packages(),
    # packages=['dyco'],
    version='1.0.1',
    license='GNU General Public License v3 (GPLv3)',
    description='A Python package to detect and compensate for shifting lag times in ecosystem time series',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lukas Hörtnagl',
    author_email='lukas.hoertnagl@usys.ethz.ch',
    url='https://gitlab.ethz.ch/holukas/dyco-dynamic-lag-compensation',
    download_url='https://pypi.org/project/dyco/',
    keywords=['ecosystem', 'eddy covariance', 'fluxes',
              'time series', 'lag', 'timeshift'],
    install_requires=['pandas', 'numpy', 'matplotlib', 'scipy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
