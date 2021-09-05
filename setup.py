import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "torchbot",
    version = "0.0.1",
    author = "He Huang",
    author_email = "stevehuanghe@gmail.com",
    description = ("A simple pytorch engine for training and evaluation"),
    license = "MIT",
    keywords = "pytorch engine",
    url = "https://github.com/stevehuanghe/torch_lib",
    packages=['torchbot'],
    long_description=read('README.md'),
)