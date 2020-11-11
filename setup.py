# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os
from setuptools import setup, find_packages


try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''

# version
here = os.path.dirname(os.path.abspath(__file__))
initpath = os.path.join(here, "src", 'jos3', '__init__.py')

for line in open(initpath):
    if "version" in line:
        line = line.split('=')[1].strip().replace('"', "").replace("'", "")
version = line

setup(
    name="jos3",
    version=version,
    url='https://github.com/TanabeLab/JOS-3',
    author='Yoshito Takahashi',
    author_email='takahashiyoshito64@gmail.com',
    description='Joint-thermoregulation system, JOS-3',
    long_description=readme,
    packages=find_packages("src"),
    package_dir={'': "src"},
    install_requires=["numpy"],
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
    ],)
