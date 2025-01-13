from setuptools import find_packages,setup
from typing import List

end= '-e .'

def get_requirements(file_path:str)->List[str]:

    requirements=[]
    with open(file_path) as file_object:
        requirements=file_object.readlines()
        requirements=[req.replace("\n","") for req in requirements]


    if end in requirements:
        requirements.remove(end)
    return requirements


setup(
    Name='LOAN_ELIGIBILITY',
    Version="0.0.1",
    Author='Sarah',
    Author_Email='ugbahsarah1999@gmail.com',
    Packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)