from setuptools import setup, find_packages

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='sar',
    version='0.1.0',
    python_requires='>=3.8',
    install_requires=[
        'dgl>=1.0.0',
        'numpy>=1.22.0',
        'torch>=1.10.0',
        'ifaddr>=0.1.7',
        'packaging>=23.1'
    ],
    packages=find_packages(),
    author='Hesham Mostafa',
    author_email='hesham.mostafa@intel.com',
    maintainer='Kacper Pietkun',
    maintainer_email='kacper.pietkun@intel.com',
    description='A Python library for distributed training of Graph Neural Networks (GNNs) on large graphs, '
                'supporting both full-batch and sampling-based training, and utilizing a sequential aggregation'
                'and rematerialization technique for linear memory scaling.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    project_urls={
        'GitHub': 'https://github.com/IntelLabs/SAR/',
        'Documentation': 'https://sar.readthedocs.io/en/latest/',
    },
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8"
    ]
)