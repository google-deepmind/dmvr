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

"""Tests for utils."""

from dmvr import utils
import tensorflow as tf


class UtilsTest(tf.test.TestCase):

  def test_combine_datasets(self):
    ds_0 = tf.data.Dataset.from_tensor_slices({
        'feature_0': [[[0] * 10] * 10] * 5,
        'feature_1': [[0] * 10] * 5,
    })
    ds_1 = tf.data.Dataset.from_tensor_slices({
        'feature_0': [[[1] * 10] * 10] * 5,
        'feature_1': [[1] * 10] * 5,
    })
    ds_2 = tf.data.Dataset.from_tensor_slices({
        'feature_0': [[[2] * 10] * 10] * 5,
        'feature_1': [[2] * 10] * 5,
    })

    # Dataset uniformly sampling from all 3 datasets.
    ds_uniform = utils.combine_datasets([ds_0, ds_1, ds_2], 7)
    data_uniform = next(iter(ds_uniform))

    # Dataset sampling from ds_1 and ds_2.
    ds_no_1 = utils.combine_datasets([ds_0, ds_1, ds_2], 7, [0.5, 0, 0.5])
    data_no_1 = next(iter(ds_no_1))

    self.assertSetEqual(set(data_uniform.keys()),
                        set(['feature_0', 'feature_1']))
    self.assertAllEqual(data_uniform['feature_0'].shape, (7, 10))
    self.assertAllEqual(data_uniform['feature_1'].shape, (7,))

    self.assertSetEqual(set(data_no_1.keys()),
                        set(['feature_0', 'feature_1']))
    self.assertAllEqual(data_no_1['feature_0'].shape, (7, 10))
    self.assertAllEqual(data_no_1['feature_1'].shape, (7,))

    self.assertAllInSet(data_uniform['feature_0'], (0, 1, 2))
    self.assertAllInSet(data_uniform['feature_1'], (0, 1, 2))
    self.assertAllInSet(data_no_1['feature_0'], (0, 2))
    self.assertAllInSet(data_no_1['feature_1'], (0, 2))


if __name__ == '__main__':
  tf.test.main()
