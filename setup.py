from setuptools import setup, find_packages

setup(
    name='s04utils',
    version='0.1.15',
    author='Janosch Kappel',
    author_email='jkl453@posteo.de',
    description='A module to load and analyze timestamp data from HDF5-photon files',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/JKL453/s04-utils', 
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    dependency_links = ['https://github.com/JKL453/sfHMM.git@main#egg=sfHMM'],
    python_requires='>=3.6',
    install_requires=[
        'h5py',
        'numpy',
        'scipy==1.11.2',
        'pandas',
        'matplotlib',
        'scikit-image',
        'pillow',
        'tabulate',
        'sfHMM',
        'bokeh',
        'pybaselines',
        'scipy',
        'seaborn',
        'fpdf',
        'tqdm',
        'reportlab'
    ]
)