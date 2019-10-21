#!/usr/bin/env python3

from setuptools import setup

MAJOR = 0
MINOR = 0
TINY  = 1
version='%d.%d.%d' % (MAJOR, MINOR, TINY)


setup(name='lincov',
      version=version,
      description='Linear covariance analysis library for lunar missions',
      author='John O. Woods, Ph.D.',
      author_email='john@openlunar.org',
      url='http://www.openlunar.org',
      include_package_data=True,
      packages=['lincov', 'test'],
      test_suite='test.lincov_test_suite'
      )
