#!/usr/bin/env python
# coding: utf-8

import setuptools

# read the contents of your README file
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setuptools.setup(
    name='bearing-python',
    version='1.0.4',
    author='Xu Fangzheng',
    author_email='fonderxu@163.com',
    url='https://github.com/fonderxu/Bearing-fault-diagnosis-datasets',
    description=u'README.md',
    install_requires=[],
    entry_points={
        'console_scripts': []
    },
    readme="README.md",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_namespace_packages(include=["bearing", "bearing.*"], ),
    include_package_data=True
)


