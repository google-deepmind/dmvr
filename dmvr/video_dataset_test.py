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

"""Tests for video_dataset."""

import os
from typing import List, Union

from dmvr import builders
from dmvr import sources
from dmvr import video_dataset
from parameterized import parameterized
import tensorflow as tf


class _TestTFRecordsSource(sources.Source):

  def load_and_decode_shard(self,
                            shard: Union[str, tf.Tensor]) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(shard)
    ds = ds.map(lambda example: (b'test_key', example))
    return ds


class _TestVideoDatasetFactory(
    video_dataset.BaseVideoDatasetFactory):

  def __init__(self, shards: List[str]):
    super().__init__(shards, builders.SequenceExampleParserBuilder,
                     _TestTFRecordsSource())

  def _build(self,
             sample_offset: int = 0,
             multiply_by_2: bool = False,
             reduce_max: bool = False,
             keep_idx: bool = False):
    self.parser_builder.parse_feature(
        'sequence', tf.io.FixedLenSequenceFeature((), dtype=tf.int64))
    if keep_idx:
      self.parser_builder.parse_feature(
          'idx', tf.io.FixedLenFeature((), dtype=tf.int64), is_context=True)

    self.sampler_builder.add_fn(
        lambda x: x[sample_offset:(sample_offset + 50)], 'sequence')

    self.decoder_builder.add_fn(
        lambda x: tf.cast(x, tf.uint8), 'sequence')

    if multiply_by_2:
      self.preprocessor_builder.add_fn(lambda x: 2 * x, 'sequence')

    if reduce_max:
      self.postprocessor_builder.add_fn(
          lambda x: tf.reduce_max(input_tensor=x, axis=1), 'sequence')


class BaseVideoDatasetFactoryTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    shards = []
    tmp_dir = self.get_temp_dir()
    # Generate TFRecords of 5 shards with serialized SequenceExamples in the
    # format ('sequence', [[0], [1], ..., [99]]) plus the shard and element
    # indices.
    for shard_idx in range(5):
      shard = os.path.join(tmp_dir,
                           'example-{:05}-of-00005.tfrecord'.format(shard_idx))
      shards.append(shard)

      # Create fake `tf.train.SequenceExample`.
      seq_example = tf.train.SequenceExample()
      for i in range(100):
        seq_example.feature_lists.feature_list.get_or_create(
            'sequence').feature.add().int64_list.value[:] = [i]

      with tf.io.TFRecordWriter(shard) as builder:
        for idx in range(10):
          seq_example.context.feature.get_or_create(
              'idx').int64_list.value[:] = [shard_idx * 10 + idx]
          builder.write(seq_example.SerializeToString())

    self._factory = _TestVideoDatasetFactory(shards)

  def test_basic(self):
    ds = self._factory.configure().make_dataset(batch_size=2)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()), set(['sequence']))
    self.assertAllEqual(data['sequence'], [list(range(50))] * 2)

  def test_configure(self):
    ds = self._factory.configure(10, True, True).make_dataset(batch_size=2)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()), set(['sequence']))
    self.assertAllEqual(data['sequence'], [59 * 2] * 2)

  def test_configure_exception(self):
    with self.assertRaises(ValueError) as _:
      self._factory.make_dataset(batch_size=2)

    with self.assertRaises(ValueError) as _:
      self._factory.configure().configure()

  def test_keep_key(self):
    ds = self._factory.configure().make_dataset(batch_size=2, keep_key=True)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()),
                        set(['sequence', builders.KEY_FEATURE_NAME]))
    self.assertAllEqual(data[builders.KEY_FEATURE_NAME].shape, (2,))
    self.assertEqual(data[builders.KEY_FEATURE_NAME][0].numpy(), b'test_key')
    self.assertEqual(data[builders.KEY_FEATURE_NAME][1].numpy(), b'test_key')

  def test_override_preprocess_fn(self):
    # Data shouldn't be multiplied by 2.
    ds = self._factory.configure(multiply_by_2=True).make_dataset(
        batch_size=2, override_preprocess_fn=lambda x: x)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()), set(['sequence']))
    self.assertAllEqual(data['sequence'], [list(range(50))] * 2)

  def test_no_shuffle(self):
    # Set block_length to guarantee reading all examples from the first shard.
    ds = self._factory.configure(keep_idx=True).tune(
        block_length=5).make_dataset(shuffle=False, batch_size=5)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()), set(['sequence', 'idx']))
    self.assertAllEqual(data['idx'], [0, 1, 2, 3, 4])

  def test_filter_read(self):
    self._factory.filter_builder.add_filter_fn(
        lambda fd: tf.not_equal(fd[builders.KEY_FEATURE_NAME], 'test_key'),
        builders.Phase.READ)
    ds = self._factory.configure().make_dataset(batch_size=10, keep_key=True)

    with self.assertRaises(StopIteration) as _:
      next(iter(ds))

  @parameterized.expand(
      ((builders.Phase.PARSE,), (builders.Phase.SAMPLE,),
       (builders.Phase.DECODE,), (builders.Phase.PREPROCESS,)))
  def test_filter(self, phase):

    def keep_even_idx(features_dict):
      idx = features_dict['idx']
      return tf.equal(idx % 2, 0)

    self._factory.filter_builder.add_filter_fn(keep_even_idx, phase)
    # Set block_length to guarantee reading examples in key order.
    ds = self._factory.configure(keep_idx=True).tune(
        block_length=10).make_dataset(shuffle=False, batch_size=10)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()), set(['sequence', 'idx']))
    self.assertAllEqual(data['idx'], range(0, 20, 2))

  def test_filter_postprocess(self):
    self._factory.filter_builder.add_filter_fn(
        lambda fd: tf.not_equal(fd['idx'][0], 0),  # Filter first batch.
        builders.Phase.POSTPROCESS)
    # Set block_length to guarantee reading examples in key order.
    ds = self._factory.configure(keep_idx=True).tune(
        block_length=10).make_dataset(shuffle=False, batch_size=10)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()), set(['sequence', 'idx']))
    self.assertAllEqual(data['idx'], range(10, 20))

  def test_ignore_processing_errors(self):

    def fail_decode(idx):
      # Fail for all odd indices.
      error = tf.assert_equal(idx % 2, tf.zeros((), dtype=tf.int64))
      with tf.control_dependencies([error]):
        return idx

    self._factory.decoder_builder.add_fn(fail_decode, 'idx')
    # Set block_length to guarantee reading examples in key order.
    ds = self._factory.configure(keep_idx=True).tune(
        block_length=10).make_dataset(
            shuffle=False, batch_size=10, ignore_processing_errors=True)

    data = next(iter(ds))
    self.assertSetEqual(set(data.keys()), set(['sequence', 'idx']))
    self.assertAllEqual(data['idx'], range(0, 20, 2))


if __name__ == '__main__':
  tf.test.main()
