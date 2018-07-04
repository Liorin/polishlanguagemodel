from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'tensor2tensor'
]

setup(
    name='polish_language_model_poleval',
    version='0.1',
    author = 'Adam Witkowski',
    author_email = 'adamwojciech.witkowski@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Polish language model based on data from PolEval 2018',
    requires=[]
)
