# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

import setuptools

_VERSION = '0.0.1'


def _parse_requirements(requirements_txt_path):
  parse_line = lambda l: l.split('#')[0].strip()
  with open(requirements_txt_path) as f:
    return [parse_line(l) for l in f]


setuptools.setup(
    name='dmvr',
    version=_VERSION,
    url='https://github.com/deepmind/dmvr',
    license='Apache 2.0',
    author='DeepMind',
    description=(
        'DMVR is a library for reading and processing multimodal datasets.'),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='dmvr-dev-os@google.com',
    # Contained modules and scripts.
    packages=setuptools.find_namespace_packages(exclude=['*_test.py']),
    install_requires=_parse_requirements('requirements.txt'),
    tests_require=_parse_requirements('requirements-test.txt'),
    requires_python='>=3.6',
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
)
