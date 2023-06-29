from setuptools import setup, find_packages

setup(
    name='SAR',
    version='0.1.0',
    install_requires=[
        'dgl>=1.0.0',
        'numpy>=1.22.0',
        'torch>=1.10.0',
        'ifaddr>=0.1.7'
    ],
    packages=find_packages(),
    author='Hesham Mostafa',
    author_email='hesham.mostafa@intel.com',
    maintainer='Bartlomiej Gawrych, Kacper Pietkun',
    maintainer_email='gawrych.bartlomiej@gmail.com, kacper.pietkun@intel.com',
    description='A Python library for distributed training of Graph Neural Networks (GNNs) on large graphs, '
                'supporting both full-batch and sampling-based training, and utilizing a sequential aggregation'
                'and rematerialization technique for linear memory scaling.',
    url='https://github.com/IntelLabs/SAR/',
)