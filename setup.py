import setuptools

setuptools.setup(
      name='cohere_core',
      author = 'Barbara Frosik, Ross Harder',
      author_email = 'bfrosik@anl.gov',
      url='https://github.com/advancedPhotonSource/cohere',
      version='3.1',
      packages=setuptools.find_packages(),
      install_requires=['numpy',
                        'tensorflow',
                        'scikit-learn',
                        'tifffile',
                        ],
      classifiers=[
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
      ],

)
