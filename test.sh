#!/bin/bash

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

# Pip installs the relevant dependencies and runs DMVR tests.

set -e
set -x

python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
virtualenv -p python3 .
source bin/activate
python3 --version

# Run setup.py, install dependencies first to use pip install.
python3 -m pip install -r requirements.txt
python3 setup.py install

# Python test dependencies.
python3 -m pip install -r requirements-test.txt

# Run all tests.
python3 -m unittest discover -s 'dmvr' -p '*_test.py'

deactivate
