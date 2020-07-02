from distutils.core import setup
setup(
  name = 'dyla',         # How you named your package folder (MyLib)
  packages = ['dyla'],   # Chose the same as "name"
  version = '0.0.1',      # Start with a small number and increase it with every change you make
  license='GNU General Public License v3.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A Python package to detect and compensate for shifting lag times in ecosystem time series',   # Give a short description about your library
  author = 'Lukas Hörtnagl',                   # Type in your name
  author_email = 'lukas.hoertnagl@usys.ethz.ch',      # Type in your E-Mail
  url = 'https://gitlab.ethz.ch/holukas/dlr-dynamic-lag-remover',   # Provide either the link to your github or to your website
  download_url = 'XXX',    # todo
  keywords = ['ecosystem', 'eddy covariance', 'fluxes',
              'time series', 'lag', 'timeshift'],   # Keywords that define your package best
  install_requires=[
          'pandas', 'numpy', 'matplotlib',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: GNU General Public License v3.0',   # Again, pick a license
    'Programming Language :: Python :: 3.6',      #Specify which pyhton versions that you want to support
  ],
)