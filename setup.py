from setuptools import find_packages,setup
from typing import List


def get_requirements(file_path:str)-> List[str]:
    requirements =[]
    with open(file_path,"r") as f:
        requirements = f.readlines()
        requirements = [req.replace('\n',"") for req in requirements]
    return requirements


setup(
    name='Sntiment_Analysis',
    version='0.0.1',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)