from setuptools import (
    find_packages,
    setup
    )
from typing import List


HYPHEN_E_DOT = "-e ."

def get_packages(file_path:str)-> List[str]:
    """Returns a list of required packages in the given requirements.txt file without the trailing '-e.' marker."""

    with open(file_path, 'r') as f:
        requirements = f.read().splitlines()

        if requirements and HYPHEN_E_DOT.strip() in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name='customer sentiment prediction',
    version='0.0.1',
    description='An ML prediction model to predict the review score for the next purchase',
    author='Derrick Njobuenwu',
    author_email='donadviser@gmail.com',
    url='',
    packages=find_packages(),
    install_requires=get_packages('requirements.txt'),
)