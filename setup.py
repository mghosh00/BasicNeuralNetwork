#
# neural_network setuptools script
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the neural_network module.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('neural_network'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_requirements():
    requirements = []
    with open("requirements.txt", "r") as file:
        for line in file:
            requirements.append(line)
    return requirements


setup(
    # Module name
    name='neural_network',

    # Version
    version=get_version(),

    description='A base package for a neural network',

    maintainer='Matthew Ghosh',

    maintainer_email='matthew.ghosh@gtc.ox.ac.uk',

    url='https://github.com/mghosh00/BasicNeuralNetwork',

    # Packages to include
    packages=find_packages(include=('neural_network', 'neural_network.*')),

    # List of dependencies
    install_requires=get_requirements(),

    extras_require={
        'docs': [
            'sphinx>=1.5, !=1.7.3',
        ],
        'dev': [
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)
