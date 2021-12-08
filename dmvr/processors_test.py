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

"""Tests for processors."""

import itertools
import os

from absl.testing import parameterized
from dmvr import processors
from dmvr import tokenizers
import numpy as np
import tensorflow as tf

# Removed: Internal pyglib dependencies

_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')
_SAMPLE_IMAGE_PATH = os.path.join(_TESTDATA_DIR, 'sample.jpeg')
_VOCAB_PATH = os.path.join(_TESTDATA_DIR, 'tokenizers', 'word_vocab.txt')


class SampleTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._sequence = tf.range(100)

  def test_sample_linspace_sequence(self):
    sampled_seq_1 = processors.sample_linspace_sequence(self._sequence, 10, 10)
    sampled_seq_2 = processors.sample_linspace_sequence(self._sequence, 7, 10)
    sampled_seq_3 = processors.sample_linspace_sequence(self._sequence, 7, 5, 2)
    sampled_seq_4 = processors.sample_linspace_sequence(self._sequence, 101, 1)
    self.assertAllEqual(sampled_seq_1, range(100))
    # [0, 1, 2, 3, 4, ..., 8, 9, 15, 16, ..., 97, 98, 99]
    self.assertAllEqual(
        sampled_seq_2,
        [15 * i + j for i, j in itertools.product(range(7), range(10))])
    # [0, 2, 4, 6, 8, 15, 17, 19, ..., 96, 98]
    self.assertAllEqual(
        sampled_seq_3,
        [15 * i + 2 * j for i, j in itertools.product(range(7), range(5))])
    self.assertAllEqual(sampled_seq_4, [0] + list(range(100)))

  def test_sample_sequence(self):
    sampled_seq_1 = processors.sample_sequence(self._sequence, 10, False)
    sampled_seq_2 = processors.sample_sequence(self._sequence, 10, False, 2)
    sampled_seq_3 = processors.sample_sequence(self._sequence, 10, True)

    self.assertAllEqual(sampled_seq_1, range(45, 55))
    self.assertAllEqual(sampled_seq_2, range(40, 60, 2))

    offset_3 = sampled_seq_3[0]
    self.assertBetween(offset_3, 0, 99)
    self.assertAllEqual(sampled_seq_3, range(offset_3, offset_3 + 10))

  def test_sample_sequence_with_state(self):
    state = {}
    sampled_seq_1 = processors.sample_sequence(
        self._sequence, 10, True, state=state)
    sampled_seq_2 = processors.sample_sequence(
        self._sequence, 10, True, state=state)

    self.assertAllEqual(sampled_seq_1, sampled_seq_2)

  def test_sample_or_pad_non_sorted_sequence(self):
    sampled_seq_1 = processors.sample_or_pad_non_sorted_sequence(
        self._sequence, 10, 0, False)
    sampled_seq_2 = processors.sample_or_pad_non_sorted_sequence(
        self._sequence, 110, 0, False)

    self.assertAllEqual(sampled_seq_1, range(10))
    self.assertAllEqual(sampled_seq_2, list(range(100)) + [0] * 10)

  def test_sample_or_pad_non_sorted_sequence_with_state(self):
    state = {}
    sampled_seq_1 = processors.sample_or_pad_non_sorted_sequence(
        self._sequence, 10, 0, True, state=state)
    sampled_seq_2 = processors.sample_or_pad_non_sorted_sequence(
        self._sequence, 10, 0, True, state=state)

    self.assertAllEqual(sampled_seq_1, sampled_seq_2)
    self.assertRaises(
        tf.errors.InvalidArgumentError,
        processors.sample_or_pad_non_sorted_sequence,
        self._sequence[:10], 10, 0, True, state=state)

  def test_sample_or_pad_non_sorted_sequence_multidim_with_state(self):
    state = {}
    sampled_seq_1 = processors.sample_or_pad_non_sorted_sequence(
        self._sequence, 10, 0, True, state=state)
    multi_dim_sequence = tf.tile(self._sequence[:, None], (1, 10))
    sampled_seq_2 = processors.sample_or_pad_non_sorted_sequence(
        multi_dim_sequence, 10, 0, True, state=state)
    self.assertAllEqual(sampled_seq_1, sampled_seq_2[:, 0])

  @parameterized.named_parameters(
      {
          'testcase_name': 'len(seq) < num_steps',
          'sequence': np.array([1, 2, 3]),
          'num_steps': 5,
          'expected_sequence': np.array([1, 2, 3, 1, 2])
      },
      {
          'testcase_name': 'len(seq) == num_steps',
          'sequence': np.array([1, 2, 3]),
          'num_steps': 3,
          'expected_sequence': np.array([1, 2, 3])
      },
      {
          'testcase_name': 'len(seq) < num_steps with stride',
          'sequence': np.array([1, 2, 3]),
          'num_steps': 5,
          'expected_sequence': np.array([1, 3, 2, 1, 3]),
          'stride': 2
      },
      {
          'testcase_name': 'len(seq) == num_steps with stride',
          'sequence': np.array([1, 2, 3]),
          'num_steps': 3,
          'expected_sequence': np.array([1, 1, 1]),
          'stride': 3
      },
  )
  def test_sample_sequence_fixed_offset(self,
                                        sequence: np.ndarray,
                                        num_steps: int,
                                        expected_sequence: np.ndarray,
                                        stride: int = 1):
    """Tests that offset is always 0."""
    for seed in range(5):
      actual_sequence = processors.sample_sequence(
          sequence, num_steps=num_steps, random=True, stride=stride, seed=seed)
      np.testing.assert_array_equal(actual_sequence, expected_sequence)


class DecodeTest(tf.test.TestCase):

  def test_decode_jpeg(self):
    with open(_SAMPLE_IMAGE_PATH, 'rb') as f: raw_image_bytes = f.read()
    raw_image = tf.constant([raw_image_bytes, raw_image_bytes])
    decoded_image = processors.decode_jpeg(raw_image)
    decoded_image_with_static_channels = processors.decode_jpeg(raw_image, 3)
    self.assertEqual(decoded_image_with_static_channels.shape.as_list()[3], 3)
    self.assertAllEqual(decoded_image.shape, (2, 263, 320, 3))
    self.assertAllEqual(decoded_image_with_static_channels.shape,
                        (2, 263, 320, 3))


class PreprocessTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # [[0, 1, ..., 119], [1, 2, ..., 120], ..., [119, 120, ..., 218]].
    self._frames = tf.stack([tf.range(i, i + 120) for i in range(90)])
    self._frames = tf.cast(self._frames, tf.uint8)
    self._frames = self._frames[tf.newaxis, :, :, tf.newaxis]
    self._frames = tf.broadcast_to(self._frames, (6, 90, 120, 3))

    # Create an equivalent numpy array for assertions.
    self._np_frames = np.array([range(i, i + 120) for i in range(90)])
    self._np_frames = self._np_frames[np.newaxis, :, :, np.newaxis]
    self._np_frames = np.broadcast_to(self._np_frames, (6, 90, 120, 3))

  def test_set_shape(self):
    with open(_SAMPLE_IMAGE_PATH, 'rb') as f: raw_image = f.read()
    raw_image = tf.constant([raw_image])
    decoded_image = processors.decode_jpeg(raw_image)
    decoded_image = processors.set_shape(decoded_image, (1, 263, 320, 3))
    self.assertAllEqual(decoded_image.shape.as_list(), (1, 263, 320, 3))

  def test_crop_image(self):
    cropped_image_1 = processors.crop_image(self._frames, 50, 70)
    cropped_image_2 = processors.crop_image(self._frames, 200, 200)
    cropped_image_3 = processors.crop_image(self._frames, 50, 70, True)

    self.assertAllEqual(cropped_image_1.shape, (6, 50, 70, 3))
    self.assertAllEqual(cropped_image_1, self._np_frames[:, 20:70, 25:95, :])

    self.assertAllEqual(cropped_image_2.shape, (6, 200, 200, 3))
    expected = np.pad(
        self._np_frames, ((0, 0), (55, 55), (40, 40), (0, 0)), 'constant')
    self.assertAllEqual(cropped_image_2, expected)

    self.assertAllEqual(cropped_image_3.shape, (6, 50, 70, 3))
    offset = cropped_image_3[0, 0, 0, 0]
    expected = np.array([range(i, i + 70) for i in range(offset, offset + 50)])
    expected = expected[np.newaxis, :, :, np.newaxis]
    expected = np.broadcast_to(expected, (6, 50, 70, 3))
    self.assertAllEqual(cropped_image_3, expected)

  def test_crop_image_with_state(self):
    state = {}
    cropped_image_1 = processors.crop_image(self._frames, 50, 70, state=state)
    cropped_image_2 = processors.crop_image(self._frames, 50, 70, state=state)

    self.assertAllEqual(cropped_image_1, cropped_image_2)

  def test_resize_smallest(self):
    resized_frames_1 = processors.resize_smallest(self._frames, 180)
    resized_frames_2 = processors.resize_smallest(self._frames, 45)
    resized_frames_3 = processors.resize_smallest(self._frames, 90)
    resized_frames_4 = processors.resize_smallest(
        tf.transpose(a=self._frames, perm=(0, 2, 1, 3)), 45)
    self.assertAllEqual(resized_frames_1.shape, (6, 180, 240, 3))
    self.assertAllEqual(resized_frames_2.shape, (6, 45, 60, 3))
    self.assertAllEqual(resized_frames_3.shape, (6, 90, 120, 3))
    self.assertAllEqual(resized_frames_4.shape, (6, 60, 45, 3))

  def test_resize_smallest_with_flow(self):
    flows = tf.cast(self._frames, tf.float32)
    resized_flows = processors.resize_smallest(flows, 180, True)
    resized_flows_expected = 2.0 * processors.resize_smallest(flows, 180, False)

    self.assertAllEqual(resized_flows, resized_flows_expected)

  def test_random_flip_left_right(self):
    flipped_frames = processors.random_flip_left_right(self._frames)
    flipped = np.fliplr(self._np_frames[0, :, :, 0])
    flipped = flipped[np.newaxis, :, :, np.newaxis]
    flipped = np.broadcast_to(flipped, (6, 90, 120, 3))
    self.assertTrue((flipped_frames == self._np_frames).numpy().all() or (
        flipped_frames == flipped).numpy().all())

  def test_random_flip_left_right_with_flow(self):
    flows = tf.cast(self._frames, tf.float32)
    flipped_flows = processors.random_flip_left_right(flows, is_flow=True)
    flipped = np.fliplr(self._np_frames[0, :, :, 0])
    flipped = flipped[np.newaxis, :, :, np.newaxis]
    flipped = np.broadcast_to(flipped, (6, 90, 120, 3))
    flipped_flow = flipped.astype(np.float32)
    flipped_flow[:, :, :, 0] *= -1.0
    self.assertTrue(
        (flipped_flows == self._np_frames.astype(np.float32)).numpy().all() or (
            flipped_flows == flipped_flow).numpy().all())

  def test_random_flip_left_right_with_state(self):
    state = {}
    flipped_frames_1 = processors.random_flip_left_right(
        self._frames, state=state)
    flipped_frames_2 = processors.random_flip_left_right(
        self._frames, state=state)
    self.assertAllEqual(flipped_frames_1, flipped_frames_2)

  def test_normalize_image(self):
    normalized_images_1 = processors.normalize_image(
        self._frames, False, tf.float32)
    normalized_images_2 = processors.normalize_image(
        self._frames, True, tf.float32)
    self.assertAllClose(normalized_images_1, self._np_frames / 255)
    self.assertAllClose(normalized_images_2, self._np_frames * 2 / 255 - 1.0)

  def test_scale_jitter_augm(self):
    no_jitter_images = processors.scale_jitter_augm(self._frames, 0.8, 1.0, 0.0)
    jitter_images = processors.scale_jitter_augm(
        self._frames, 2.0, 2.00001, 1.0)
    self.assertAllEqual(no_jitter_images.shape, (6, 90, 120, 3))
    self.assertAllEqual(jitter_images.shape, (6, 180, 240, 3))

  def test_scale_jitter_augm_with_state(self):
    state = {}
    jitter_image_1 = processors.scale_jitter_augm(
        self._frames, 0.8, 1.2, 1.0, state=state)
    jitter_image_2 = processors.scale_jitter_augm(
        self._frames, 0.8, 1.2, 1.0, state=state)
    self.assertAllEqual(jitter_image_1, jitter_image_2)

  def test_scale_jitter_augm_with_flow(self):
    state = {}
    flows = tf.cast(self._frames, tf.float32)
    jitter_flows = processors.scale_jitter_augm(
        flows, 0.8, 1.2, 1.0, state=state, is_flow=True)
    jitter_flows_expected = processors.scale_jitter_augm(
        flows, 0.8, 1.2, 1.0, state=state)
    h_s, w_s, _ = state['scale_jitter_augm_info']
    jitter_flows_expected *= tf.stack([h_s, w_s, 1.0])[None, None, None, :]
    self.assertAllClose(jitter_flows, jitter_flows_expected)

  def test_color_default_augment(self):
    normalized_images = processors.normalize_image(
        self._frames, False, tf.float32)
    no_augmented_images = processors.color_default_augm(
        normalized_images, False, 0.0, 0.0)
    color_augmented_images = processors.color_default_augm(
        normalized_images, False, 1.0, 0.0)
    color_dropped_images = processors.color_default_augm(
        normalized_images, False, 0.0, 1.0)
    self.assertAllEqual(no_augmented_images.shape, normalized_images.shape)
    self.assertAllEqual(color_augmented_images.shape, normalized_images.shape)
    self.assertAllEqual(color_dropped_images.shape, normalized_images.shape)

    self.assertAllEqual(normalized_images, no_augmented_images)
    self.assertNotAllEqual(normalized_images, color_augmented_images)
    self.assertNotAllEqual(normalized_images, color_dropped_images)

    self.assertAllEqual(color_dropped_images[:, :, :, 0],
                        color_dropped_images[:, :, :, 1])
    self.assertAllEqual(color_dropped_images[:, :, :, 0],
                        color_dropped_images[:, :, :, 2])

  def test_space_to_depth(self):
    output_frames_1 = processors.space_to_depth(self._frames, 2, 3)
    output_frames_2 = processors.space_to_depth(self._frames, 3, 2)
    output_frames_3 = processors.space_to_depth(
        self._frames, spatial_block_size=2)
    self.assertAllEqual(output_frames_1.shape, (3, 30, 40, 54))
    self.assertAllEqual(output_frames_2.shape, (2, 45, 60, 36))
    self.assertAllEqual(output_frames_3.shape, (6, 45, 60, 12))

  def test_crop_or_pad_words(self):
    input_words_indices = tf.expand_dims(tf.range(10, dtype=tf.int32), axis=0)

    output_words_indices_1 = processors.crop_or_pad_words(
        input_words_indices, 5)
    output_words_indices_2 = processors.crop_or_pad_words(
        input_words_indices, 15)
    self.assertAllEqual(output_words_indices_1, [list(range(5))])
    self.assertAllEqual(output_words_indices_2,
                        [[i for i in range(10)] + [0] * 5])

  def test_tokenize(self):
    tokenizer = tokenizers.WordTokenizer(
        _VOCAB_PATH)  # OSS: removed internal filename loading.
    tokenizer.initialize()
    input_features = {'text': tf.constant(['hello world', 'hello', 'world'])}

    output_features = processors.tokenize(input_features, tokenizer, 'text',
                                          'indices', False, False, 4, True)
    self.assertAllEqual(output_features['text'],
                        ['hello world', 'hello', 'world'])
    self.assertAllEqual(output_features['indices'],
                        [[4, 5, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]])


class PostprocessTest(tf.test.TestCase):

  def test_batched_video_transpose(self):
    input_tensor = tf.constant([[[1, 2], [3, 4], [5, 6]]])
    output_tensor = processors.batched_video_transpose(input_tensor, (0, 2, 1))

    self.assertAllEqual(output_tensor, [[[1, 3, 5], [2, 4, 6]]])

  def test_batched_space_to_depth(self):
    input_frames = tf.zeros((8, 30, 150, 210, 3))

    output_frames_1 = processors.batched_space_to_depth(input_frames, 2, 3)
    output_frames_2 = processors.batched_space_to_depth(input_frames, 3, 2)
    output_frames_3 = processors.batched_space_to_depth(
        input_frames, spatial_block_size=2)

    self.assertAllEqual(output_frames_1.shape, (8, 15, 50, 70, 54))
    self.assertAllEqual(output_frames_2.shape, (8, 10, 75, 105, 36))
    self.assertAllEqual(output_frames_3.shape, (8, 30, 75, 105, 12))


if __name__ == '__main__':
  tf.test.main()
