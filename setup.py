#!/usr/bin/env python

from setuptools import setup

with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f.readlines()]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='dcsem',
      description='DCM and SEM tool',
      author='Saad Jbabdi',
      author_email='saad.jbabdi@ndcn.ox.ac.uk',
      packages=['dcsem', 'dcsem/auxilary'],
      scripts=['dcsem/scripts/dcsem_sim',],
      install_requires=install_requires,
      )

