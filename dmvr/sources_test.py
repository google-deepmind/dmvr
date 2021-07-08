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

"""Tests for sources."""

import os

from dmvr import sources
import tensorflow as tf


class TFRecordsSourceTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self._shard = os.path.join(self.get_temp_dir(), 'shard')
    # Generate a TFRecord single shard with one serialized SequenceExample
    # in the format ('sequence', [[0], [1], ..., [99]]).
    with tf.io.TFRecordWriter(self._shard) as builder:
      self._seq_example = tf.train.SequenceExample()
      for i in range(100):
        self._seq_example.feature_lists.feature_list.get_or_create(
            'sequence').feature.add().int64_list.value[:] = [i]
      builder.write(self._seq_example.SerializeToString())

  def test_load_and_decode(self):
    source = sources.TFRecordsSource()
    ds = source.load_and_decode_shard(self._shard)
    it = iter(ds)

    data = next(it)
    self.assertEqual(data[0], self._shard.encode('utf-8'))
    self.assertEqual(data[1], self._seq_example.SerializeToString())

    with self.assertRaises(StopIteration) as _:
      data = next(it)

  def test_input_as_tensor(self):
    source = sources.TFRecordsSource()
    ds = source.load_and_decode_shard(tf.constant(self._shard))
    it = iter(ds)

    data = next(it)
    self.assertEqual(data[0], self._shard.encode('utf-8'))
    self.assertEqual(data[1], self._seq_example.SerializeToString())


if __name__ == '__main__':
  tf.test.main()
