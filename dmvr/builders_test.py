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

"""Tests for builders."""

from dmvr import builders
from parameterized import parameterized
import tensorflow as tf


class SequenceExampleParserBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Prepare SequenceExample.
    seq_example = tf.train.SequenceExample()
    seq_example.context.feature.get_or_create(
        'my_context_feature').int64_list.value[:] = [0, 1]
    seq_example.feature_lists.feature_list.get_or_create(
        'my_seq_feature').feature.add().int64_list.value[:] = [2, 3]
    seq_example.feature_lists.feature_list.get(
        'my_seq_feature').feature.add().int64_list.value[:] = [4, 5]
    seq_example.feature_lists.feature_list.get_or_create(
        'my_var_len_seq_feature').feature.add().int64_list.value[:] = [6]
    seq_example.feature_lists.feature_list.get(
        'my_var_len_seq_feature').feature.add().int64_list.value[:] = [7, 8]

    # Put SequenceExample in expected format.
    self._raw_seq_example = tf.constant(seq_example.SerializeToString())

  def test_parse(self):
    parse_fn = (
        builders.SequenceExampleParserBuilder()
        .parse_feature('my_context_feature',
                       tf.io.FixedLenFeature((2,), dtype=tf.int64),
                       'context_name', True)
        .parse_feature('my_seq_feature',
                       tf.io.FixedLenSequenceFeature((2,), dtype=tf.int64),
                       'seq_name')
        .parse_feature('my_var_len_seq_feature',
                       tf.io.VarLenFeature(dtype=tf.int64),
                       'var_len_seq_name')
        .build())
    features_dict = parse_fn(self._raw_seq_example)

    self.assertSetEqual(set(['context_name', 'seq_name', 'var_len_seq_name']),
                        set(features_dict.keys()))
    self.assertAllEqual(features_dict['context_name'], [0, 1])
    self.assertAllEqual(features_dict['seq_name'], [[2, 3], [4, 5]])
    self.assertAllEqual(features_dict['var_len_seq_name'].values, [6, 7, 8])
    self.assertAllEqual(features_dict['var_len_seq_name'].indices,
                        [[0, 0], [1, 0], [1, 1]])
    self.assertAllEqual(features_dict['var_len_seq_name'].dense_shape, [2, 2])

  def test_fake_data(self):
    parser = builders.SequenceExampleParserBuilder()
    parser.parse_feature('my_context_feature',
                         tf.io.FixedLenFeature((2,), dtype=tf.int64),
                         'context_name', True)
    parser.parse_feature('my_seq_feature',
                         tf.io.FixedLenSequenceFeature((2,), dtype=tf.int64),
                         'seq_name')
    parser.parse_feature('my_var_len_seq_feature',
                         tf.io.VarLenFeature(dtype=tf.int64),
                         'var_len_seq_name')
    fake_data = parser.get_fake_data(default_values={
        'context_name': (0, 1), 'var_len_seq_name': ((1, 2), (3, 4, 5))})
    self.assertSetEqual(set(['context_name', 'seq_name', 'var_len_seq_name']),
                        set(fake_data.keys()))
    self.assertAllEqual(fake_data['context_name'], [0, 1])
    self.assertAllEqual(fake_data['seq_name'], [[0, 0]])
    self.assertAllEqual(fake_data['var_len_seq_name'].values, [1, 2, 3, 4, 5])
    self.assertAllEqual(fake_data['var_len_seq_name'].dense_shape, [2, 3])

  def test_no_output_name(self):
    parse_fn = (
        builders.SequenceExampleParserBuilder()
        .parse_feature('my_context_feature',
                       tf.io.FixedLenFeature((2,), dtype=tf.int64),
                       is_context=True)
        .build())
    features_dict = parse_fn(self._raw_seq_example)

    self.assertSetEqual(set(['my_context_feature']), set(features_dict.keys()))

  def test_same_output_name(self):
    parser_builder = builders.SequenceExampleParserBuilder()
    parser_builder.parse_feature('my_context_feature',
                                 tf.io.FixedLenFeature((2,), dtype=tf.int64),
                                 'same_name', True)
    parser_builder.parse_feature('my_context_feature',
                                 tf.io.FixedLenFeature((2,), dtype=tf.int64),
                                 'other_name', True)

    with self.assertRaises(ValueError) as _:
      parser_builder.parse_feature(
          'my_seq_feature', tf.io.FixedLenSequenceFeature((2,), dtype=tf.int64),
          'same_name')

  def test_different_types_for_same_feature(self):
    parser_builder = builders.SequenceExampleParserBuilder()
    parser_builder.parse_feature('my_context_feature',
                                 tf.io.FixedLenFeature((2,), dtype=tf.int64),
                                 'context_name', True)

    with self.assertRaises(ValueError) as _:
      parser_builder.parse_feature('my_context_feature',
                                   tf.io.FixedLenFeature((3,), dtype=tf.int64),
                                   'context_name_2', True)

    with self.assertRaises(ValueError) as _:
      parser_builder.parse_feature('my_context_feature',
                                   tf.io.FixedLenFeature((2,), dtype=tf.string),
                                   'context_name_3', True)


class ExampleParserBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Prepare Example.
    tf_example = tf.train.Example()
    tf_example.features.feature.get_or_create(
        'my_fixed_len_feature').int64_list.value[:] = [0, 1]
    tf_example.features.feature.get_or_create(
        'my_var_len_feature').int64_list.value[:] = [2, 3, 4]

    # Put Example in expected format.
    self._raw_tf_example = tf.constant(tf_example.SerializeToString())

  def test_parse(self):
    parse_fn = (
        builders.ExampleParserBuilder()
        .parse_feature('my_fixed_len_feature',
                       tf.io.FixedLenFeature((2,), dtype=tf.int64),
                       'fixed_name')
        .parse_feature('my_var_len_feature',
                       tf.io.VarLenFeature(dtype=tf.int64), 'var_name')
        .build())
    features_dict = parse_fn(self._raw_tf_example)

    self.assertSetEqual(set(['fixed_name', 'var_name']),
                        set(features_dict.keys()))
    self.assertAllEqual(features_dict['fixed_name'], [0, 1])
    self.assertAllEqual(features_dict['var_name'].values, [2, 3, 4])
    self.assertAllEqual(features_dict['var_name'].indices, [[0], [1], [2]])
    self.assertAllEqual(features_dict['var_name'].dense_shape, [3])

  def test_fake_data(self):
    fake_data = (
        builders.ExampleParserBuilder()
        .parse_feature('my_fixed_len_feature',
                       tf.io.FixedLenFeature((2,), dtype=tf.string),
                       'fixed_name')
        .parse_feature('my_var_len_feature',
                       tf.io.VarLenFeature(dtype=tf.int64), 'var_name')
        .get_fake_data(default_values={'fixed_name': (b'42', b'25')}))
    self.assertSetEqual(set(['fixed_name', 'var_name']),
                        set(fake_data.keys()))
    self.assertAllEqual(fake_data['fixed_name'], [b'42', b'25'])
    self.assertAllEqual(fake_data['var_name'].values, [0])
    self.assertAllEqual(fake_data['var_name'].indices, [[0]])
    self.assertAllEqual(fake_data['var_name'].dense_shape, [1])

  def test_no_output_name(self):
    parse_fn = (
        builders.ExampleParserBuilder()
        .parse_feature('my_fixed_len_feature',
                       tf.io.FixedLenFeature((2,), dtype=tf.int64))
        .build())
    features_dict = parse_fn(self._raw_tf_example)

    self.assertSetEqual(set(['my_fixed_len_feature']),
                        set(features_dict.keys()))

  def test_same_output_name(self):
    parser_builder = builders.ExampleParserBuilder()
    parser_builder.parse_feature('my_fixed_len_feature',
                                 tf.io.FixedLenFeature((2,), dtype=tf.int64),
                                 'same_name')
    parser_builder.parse_feature('my_fixed_len_feature',
                                 tf.io.FixedLenFeature((2,), dtype=tf.int64),
                                 'other_name')

    with self.assertRaises(ValueError) as _:
      parser_builder.parse_feature(
          'my_var_len_feature',
          tf.io.FixedLenSequenceFeature((2,), dtype=tf.int64), 'same_name')

  def test_different_types_for_same_feature(self):
    parser_builder = builders.SequenceExampleParserBuilder()
    parser_builder.parse_feature('my_fixed_len_feature',
                                 tf.io.FixedLenFeature((2,), dtype=tf.int64),
                                 'fixed_name')

    with self.assertRaises(ValueError) as _:
      parser_builder.parse_feature('my_fixed_len_feature',
                                   tf.io.FixedLenFeature((3,), dtype=tf.int64),
                                   'fixed_name_2')

    with self.assertRaises(ValueError) as _:
      parser_builder.parse_feature('my_fixed_len_feature',
                                   tf.io.FixedLenFeature((2,), dtype=tf.string),
                                   'fixed_name_3')


def _add_one(x):
  return tf.math.add(x, 1)


def _subtract_one(x):
  return tf.math.subtract(x, 1)


def _upper_text(x):
  return tf.strings.upper(x)


def _add_text_len(features_dict):
  features_dict['feature_3'] = tf.strings.length(
      input=features_dict['feature_2'])
  return features_dict


def _set_state(x, state):
  state['value'] = x
  return x


def _use_state(features_dict, state):
  features_dict['feature_4'] = state['value']
  return features_dict


class BuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Prepare features dictionary.
    self._input_features_dict = {
        'feature_1': tf.constant(0),
        'feature_2': tf.constant('text')
    }

  def test_basic(self):
    process_fn = (
        builders._Builder()
        .add_fn(_add_one, 'feature_1')
        .add_fn(_upper_text, 'feature_2')
        .add_fn(_add_text_len)
        .add_fn(_add_one, 'feature_1')
        .build())
    output_features_dict = process_fn(self._input_features_dict)

    self.assertSetEqual(
        set(['feature_1', 'feature_2', 'feature_3']),
        set(output_features_dict.keys()))
    self.assertEqual(output_features_dict['feature_1'], 2)
    self.assertEqual(output_features_dict['feature_2'], b'TEXT')
    self.assertEqual(output_features_dict['feature_3'], 4)

  def test_replace(self):
    process_fn = (
        builders._Builder()
        .add_fn(_add_one, 'feature_1', 'add_one')
        .add_fn(_upper_text, 'feature_2')
        .replace_fn('add_one', _subtract_one)
        .build())
    output_features_dict = process_fn(self._input_features_dict)

    self.assertSetEqual(set(['feature_1', 'feature_2']),
                        set(output_features_dict.keys()))
    self.assertEqual(output_features_dict['feature_1'], -1)
    self.assertEqual(output_features_dict['feature_2'], b'TEXT')

  def test_remove(self):
    process_fn = (
        builders._Builder()
        .add_fn(_add_one, 'feature_1', 'add_one')
        .add_fn(_upper_text, 'feature_2')
        .remove_fn('add_one')
        .build())
    output_features_dict = process_fn(self._input_features_dict)

    self.assertSetEqual(set(['feature_1', 'feature_2']),
                        set(output_features_dict.keys()))
    self.assertEqual(output_features_dict['feature_1'], 0)
    self.assertEqual(output_features_dict['feature_2'], b'TEXT')

  def test_reset(self):
    process_fn = (
        builders._Builder()
        .add_fn(_add_one, 'feature_1')
        .add_fn(_upper_text, 'feature_2')
        .reset()
        .build())
    output_features_dict = process_fn(self._input_features_dict)

    self.assertSetEqual(set(['feature_1', 'feature_2']),
                        set(output_features_dict.keys()))
    self.assertEqual(output_features_dict['feature_1'], 0)
    self.assertEqual(output_features_dict['feature_2'], b'text')

  def test_stateful(self):
    process_fn = (
        builders._Builder()
        .add_fn(_set_state, 'feature_1', stateful=True)
        .add_fn(_use_state, stateful=True)
        .build())
    output_features_dict = process_fn(self._input_features_dict)

    self.assertSetEqual(set(['feature_1', 'feature_2', 'feature_4']),
                        set(output_features_dict.keys()))
    self.assertEqual(output_features_dict['feature_1'], 0)
    self.assertEqual(output_features_dict['feature_4'], 0)

  def test_same_fn_name(self):
    builder = builders._Builder().add_fn(_add_one, 'feature_1', 'add_one')

    with self.assertRaises(ValueError) as _:
      builder.add_fn(_add_one, 'feature_1', 'add_one')

  def test_replace_wrong_fn_name(self):
    builder = builders._Builder().add_fn(_add_one, 'feature_1', 'add_one')

    with self.assertRaises(ValueError) as _:
      builder.replace_fn('add_one_wrong', _add_one)

  def test_insert(self):
    def replace_string(_):
      return tf.constant('replaced_text')

    builder = builders._Builder() .add_fn(_add_text_len, fn_name='text_len')
    output_features_dict = builder.build()(self._input_features_dict)

    builder.add_fn(replace_string, 'feature_2', add_before_fn_name='text_len')
    output_features_dict_2 = builder.build()(self._input_features_dict)

    self.assertSetEqual(set(['feature_1', 'feature_2', 'feature_3']),
                        set(output_features_dict.keys()))
    self.assertEqual(output_features_dict['feature_2'], b'text')
    self.assertEqual(output_features_dict['feature_3'], 4)

    self.assertSetEqual(set(['feature_1', 'feature_2', 'feature_3']),
                        set(output_features_dict_2.keys()))
    self.assertEqual(output_features_dict_2['feature_2'], b'replaced_text')
    self.assertEqual(output_features_dict_2['feature_3'], 13)

  def test_wrong_add_before_fn_name(self):
    builder = builders._Builder().add_fn(_add_one, 'feature_1', 'add_one')

    with self.assertRaises(ValueError) as _:
      builder.add_fn(_add_one, 'feature_1', add_before_fn_name='add_one_wrong')


class FilterBuilderTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    # Prepare features dictionary.
    self._input_features_dict = {
        'feature_1': tf.constant(0),
        'feature_2': tf.constant('text'),
        'feature_3': tf.zeros((16, 200, 200, 3))
    }

  @parameterized.expand(((builders.Phase.READ,), (builders.Phase.PARSE,),
                         (builders.Phase.SAMPLE,), (builders.Phase.DECODE,),
                         (builders.Phase.PREPROCESS,),
                         (builders.Phase.POSTPROCESS,)))
  def test_drop(self, phase):
    filter_fn = (
        builders.FilterBuilder()
        .add_filter_fn(lambda fd: tf.equal(fd['feature_1'], 0), phase)
        .add_filter_fn(lambda fd: tf.equal(fd['feature_2'], 'no_text'), phase)
        .add_filter_fn(
            lambda fd: tf.equal(tf.shape(input=fd['feature_3'])[3], 3), phase)
        .build(phase))
    keep = filter_fn(self._input_features_dict)

    self.assertEqual(keep, False)

  @parameterized.expand(((builders.Phase.READ,), (builders.Phase.PARSE,),
                         (builders.Phase.SAMPLE,), (builders.Phase.DECODE,),
                         (builders.Phase.PREPROCESS,),
                         (builders.Phase.POSTPROCESS,)))
  def test_keep(self, phase):
    filter_fn = (
        builders.FilterBuilder()
        .add_filter_fn(lambda fd: tf.equal(fd['feature_1'], 0), phase)
        .add_filter_fn(lambda fd: tf.equal(fd['feature_2'], 'text'), phase)
        .add_filter_fn(
            lambda fd: tf.equal(tf.shape(input=fd['feature_3'])[3], 3), phase)
        .build(phase))
    keep = filter_fn(self._input_features_dict)

    self.assertEqual(keep, True)

  @parameterized.expand(((builders.Phase.READ,), (builders.Phase.PARSE,),
                         (builders.Phase.SAMPLE,), (builders.Phase.DECODE,),
                         (builders.Phase.PREPROCESS,),
                         (builders.Phase.POSTPROCESS,)))
  def test_empty(self, phase):
    filter_fn = builders.FilterBuilder().build(phase)
    keep = filter_fn(self._input_features_dict)

    self.assertEqual(keep, True)


if __name__ == '__main__':
  tf.test.main()
