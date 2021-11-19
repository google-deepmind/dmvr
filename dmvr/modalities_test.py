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

"""Tests for modalities."""

import os

from dmvr import builders
from dmvr import modalities
from dmvr import tokenizers
import numpy as np
from parameterized import parameterized
import tensorflow as tf

# Removed: Internal pyglib dependencies

_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')
_SAMPLE_IMAGE_PATH = os.path.join(_TESTDATA_DIR, 'sample.jpeg')
_VOCAB_PATH = os.path.join(_TESTDATA_DIR, 'tokenizers', 'word_vocab.txt')


class ModalitiesTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    seq_example = tf.train.SequenceExample()

    # Create stub frames and inject them in the SequenceExample.
    with open(_SAMPLE_IMAGE_PATH, 'rb') as f: raw_image_bytes = f.read()
    for _ in range(10 * 5):
      seq_example.feature_lists.feature_list.get_or_create(
          'image/encoded').feature.add().bytes_list.value[:] = [raw_image_bytes]

    # Create stub flow and inject it in the SequenceExample.
    for _ in range(10 * 5):
      seq_example.feature_lists.feature_list.get_or_create(
          'flow/encoded').feature.add().bytes_list.value[:] = [raw_image_bytes]

    # Create stub label and inject it in the SequenceExample.
    raw_label_index = 42
    raw_label_name = b'label'
    seq_example.context.feature.get_or_create(
        'clip/label/index').int64_list.value[:] = [raw_label_index]
    seq_example.context.feature.get_or_create(
        'clip/label/string').bytes_list.value[:] = [raw_label_name]

    # Create stub raw text and inject it in SequenceExample.
    raw_text = b'hello world'
    seq_example.context.feature.get_or_create(
        'caption/string').bytes_list.value[:] = [raw_text, raw_text]

    # Create stub audio and inject it in SequenceExample.
    raw_audio = np.linspace(-1, 1, 48000 * 5)
    seq_example.feature_lists.feature_list.get_or_create(
        'WAVEFORM/feature/floats').feature.add().float_list.value[:] = raw_audio

    serialized_seq_example = seq_example.SerializeToString()
    self._seq_examples = [serialized_seq_example] * 8  # Batch size is 8.

    # Create builders.
    self._parser_builder = builders.SequenceExampleParserBuilder()
    self._sampler_builder = builders.SamplerBuilder()
    self._decoder_builder = builders.DecoderBuilder()
    self._preprocessor_builder = builders.PreprocessorBuilder()
    self._postprocessor_builder = builders.PostprocessorBuilder()

  def _process_examples(self):
    """Process input examples simulating dataset object creation."""
    def pre_batch_process(raw_seq_example):
      output = self._parser_builder.build()(raw_seq_example)
      output = self._sampler_builder.build()(output)
      output = self._decoder_builder.build()(output)
      output = self._preprocessor_builder.build()(output)
      return output

    # Batch and postprocess.
    output = [pre_batch_process(rse) for rse in self._seq_examples]
    batched_output = {}
    for k in output[0].keys():
      batched_output[k] = tf.stack([out[k] for out in output])
    output = batched_output
    output = self._postprocessor_builder.build()(output)

    return output

  @parameterized.expand((
      (True, 1, False, True, ['image_random_sample'], [
          'image_resize_smallest', 'image_random_crop', 'image_random_flip',
          'image_normalize'
      ], []),
      (True, 1, False, False, ['image_random_sample'],
       ['image_resize_smallest', 'image_random_crop', 'image_normalize'], []),
      (False, 1, False, True, ['image_middle_sample'],
       ['image_resize_smallest', 'image_central_crop', 'image_normalize'], []),
      (False, 2, False, True, ['image_linspace_sample'],
       ['image_resize_smallest', 'image_central_crop',
        'image_normalize'], ['image_reshape']),
      (True, 1, True, True, ['image_random_sample'], [
          'image_normalize', 'image_resize_smallest', 'image_random_crop',
          'image_random_flip', 'image_extract_flow_channels', 'image_clip_flow'
      ], []),
  ))
  def test_add_image(self, is_training, num_test_clips, is_flow, random_flip,
                     sample_ops, preprocess_ops, postprocess_ops):
    is_rgb = None if is_flow else True
    zero_centering_image = is_flow
    modalities.add_image(
        self._parser_builder,  # `parser_builder`
        self._sampler_builder,  # `sampler_builder`
        self._decoder_builder,  # `decoder_builder`
        self._preprocessor_builder,  # `preprocessor_builder`
        self._postprocessor_builder,  # `postprocessor_builder`
        'image/encoded',  # `input_feature_name`
        'image',  # `output_feature_name`
        is_training,  # `is_training`
        32,  # `num_frames`
        1,  # `stride`
        num_test_clips,  # `num_test_clips`
        224,  # `min_resize`
        200,  # `crop_size`
        zero_centering_image,  # `zero_centering_image`
        True,  # `sync_random_state`
        is_rgb,  # `is_rgb`
        is_flow,  # `is_flow`
        random_flip)  # `random_flip`
    output = self._process_examples()

    self.assertAllEqual(
        [fd.fn_name for fd in self._sampler_builder.get_summary()], sample_ops)
    self.assertAllEqual(
        [fd.fn_name for fd in self._decoder_builder.get_summary()],
        ['image_decode_jpeg'])
    self.assertAllEqual(
        [fd.fn_name for fd in self._preprocessor_builder.get_summary()],
        preprocess_ops)
    self.assertAllEqual(
        [fd.fn_name for fd in self._postprocessor_builder.get_summary()],
        postprocess_ops)

    # Assert static shape.
    self.assertNotIn(None, output['image'].shape.as_list())
    self.assertSetEqual(set(output.keys()), set(['image']))
    num_output_channels = 2 if is_flow else 3
    self.assertAllEqual(output['image'].shape,
                        (8 * num_test_clips, 32, 200, 200, num_output_channels))

  @parameterized.expand(((False, False), (False, True), (True, True)))
  def test_add_label(self, one_hot_label, add_label_name):
    modalities.add_label(
        self._parser_builder,  # `parser_builder`
        self._decoder_builder,  # `decoder_builder`
        self._preprocessor_builder,  # `preprocessor_builder`
        'clip/label/index',  # `input_label_index_feature_name`
        'label',  # `output_label_index_feature_name`
        'clip/label/string',  # `input_label_name_feature_name`
        'label_name',  # `output_label_name_feature_name`
        False,  # `is_multi_label`
        one_hot_label,  # `one_hot_label`
        50,  # `num_classes`
        add_label_name)  # `add_label_name`
    output = self._process_examples()

    decoder_ops = ['label_sparse_to_dense']
    if add_label_name:
      decoder_ops.append('label_name_sparse_to_dense')
    self.assertAllEqual(
        [fd.fn_name for fd in self._decoder_builder.get_summary()],
        decoder_ops)
    if one_hot_label:
      preprocess_ops = ['label_one_hot']
    else:
      preprocess_ops = ['label_set_shape']
    if add_label_name:
      preprocess_ops.append('label_name_set_shape')
    self.assertAllEqual(
        [fd.fn_name for fd in self._preprocessor_builder.get_summary()],
        preprocess_ops)

    # Assert static shape.
    self.assertNotIn(None, output['label'].shape.as_list())

    keys = set(['label'])
    if add_label_name:
      keys.add('label_name')
    self.assertSetEqual(set(output.keys()), keys)
    if one_hot_label:
      self.assertAllEqual(output['label'], [[0] * 42 + [1] + [0] * 7] * 8)
    else:
      self.assertAllEqual(output['label'], [[42]] * 8)
    if add_label_name:
      self.assertAllEqual(output['label_name'], [[b'label']] * 8)

  @parameterized.expand(((16,), (1,)))
  def test_add_text(self, max_num_words):
    tokenizer_model = tokenizers.WordTokenizer(
        _VOCAB_PATH)  # OSS: removed internal filename loading.
    tokenizer_model.initialize()

    modalities.add_text(
        self._parser_builder,  # `parser_builder`
        self._decoder_builder,  # `decoder_builder`
        self._preprocessor_builder,  # `preprocessor_builder`
        tokenizer_model,  # `tokenizer`
        True,  # `is_training`
        'caption/string',  # `input_feature_name`
        builders.TEXT_FEATURE_NAME,  # `output_raw_name`
        builders.TEXT_INDICES_FEATURE_NAME,  # `output_feature_name`
        False,  # `prepend_bos`
        False,  # `append_eos`
        True,  # `keep_raw_string`
        2,  # `max_num_captions`
        max_num_words,  # `max_num_words`
        True)  # `sync_random_state`

    output = self._process_examples()
    self.assertAllEqual(
        [fd.fn_name for fd in self._decoder_builder.get_summary()],
        ['text_indices_sparse_to_dense'])
    self.assertAllEqual(
        [fd.fn_name for fd in self._preprocessor_builder.get_summary()],
        ['text_indices_sample_captions', 'text_indices_tokenization',
         'text_indices_set_shape'])

    # Assert static shape.
    self.assertNotIn(
        None, output[builders.TEXT_INDICES_FEATURE_NAME].shape.as_list())
    self.assertSetEqual(set(output.keys()),
                        set([builders.TEXT_INDICES_FEATURE_NAME,
                             builders.TEXT_FEATURE_NAME]))
    words = [4, 5][:min(2, max_num_words)]
    padding = [0] * max(0, max_num_words - 2)
    self.assertAllEqual(
        output[builders.TEXT_INDICES_FEATURE_NAME],
        [[words + padding, words + padding]] * 8)

  @parameterized.expand((
      (True, 1, ['audio_sparse_to_dense', 'audio_random_sample'], []),
      (False, 1, ['audio_sparse_to_dense', 'audio_middle_sample'], []),
      (False, 2, ['audio_sparse_to_dense', 'audio_linspace_sample'],
       ['audio_reshape'])))
  def test_add_audio(self, is_training, num_test_clips, sample_ops,
                     postprocess_ops):
    modalities.add_audio(
        self._parser_builder,  # `parser_builder`
        self._sampler_builder,  # `sampler_builder`
        self._postprocessor_builder,  # `postprocessor_builder`
        'WAVEFORM/feature/floats',  # `input_feature_name`
        builders.AUDIO_FEATURE_NAME,  # `output_feature_name`
        is_training,  # `is_training`
        30720,  # `num_samples`
        1,  # `stride`
        num_test_clips)  # `num_test_clips`
    output = self._process_examples()

    self.assertAllEqual(
        [fd.fn_name for fd in self._sampler_builder.get_summary()],
        sample_ops)
    self.assertAllEqual(
        [fd.fn_name for fd in self._postprocessor_builder.get_summary()],
        postprocess_ops)

    # Assert static shape.
    self.assertNotIn(
        None, output[builders.AUDIO_FEATURE_NAME].shape.as_list())
    self.assertSetEqual(set(output.keys()),
                        set([builders.AUDIO_FEATURE_NAME]))
    self.assertAllEqual(output[builders.AUDIO_FEATURE_NAME].shape,
                        (8 * num_test_clips, 30720))

  def test_all_modalities(self):
    # Add RGB image.
    modalities.add_image(self._parser_builder, self._sampler_builder,
                         self._decoder_builder, self._preprocessor_builder,
                         self._postprocessor_builder)
    # Add flow image. Note that in this test this will read from a RGB
    # flow/encoded since we store flow on disk as RGB images where only the two
    # first channels (RG) corresponds to the relevant horizontal and vertical
    # displacement vector.
    modalities.add_image(
        self._parser_builder,
        self._sampler_builder,
        self._decoder_builder,
        self._preprocessor_builder,
        self._postprocessor_builder,
        input_feature_name='flow/encoded',
        output_feature_name=builders.FLOW_FEATURE_NAME,
        is_rgb=None,
        zero_centering_image=True,
        is_flow=True)
    modalities.add_label(
        self._parser_builder,
        self._decoder_builder,
        self._preprocessor_builder,
        num_classes=50)
    tokenizer = tokenizers.WordTokenizer(
        _VOCAB_PATH)  # OSS: removed internal filename loading.
    tokenizer.initialize()
    modalities.add_text(
        self._parser_builder,
        self._decoder_builder,
        self._preprocessor_builder,
        tokenizer=tokenizer)
    modalities.add_audio(self._parser_builder, self._sampler_builder,
                         self._postprocessor_builder)
    output = self._process_examples()

    self.assertSetEqual(
        set(output.keys()),
        set([
            builders.IMAGE_FEATURE_NAME, builders.FLOW_FEATURE_NAME,
            builders.LABEL_INDEX_FEATURE_NAME,
            builders.TEXT_INDICES_FEATURE_NAME, builders.AUDIO_FEATURE_NAME
        ]))
    self.assertAllEqual(output[builders.IMAGE_FEATURE_NAME].shape,
                        (8, 32, 200, 200, 3))
    self.assertAllEqual(output[builders.FLOW_FEATURE_NAME].shape,
                        (8, 32, 200, 200, 2))
    self.assertAllEqual(output[builders.LABEL_INDEX_FEATURE_NAME],
                        [[0] * 42 + [1] + [0] * 7] * 8)
    self.assertAllEqual(output[builders.TEXT_INDICES_FEATURE_NAME],
                        [[[4, 5] + [0] * 14]] * 8)
    self.assertAllEqual(output[builders.AUDIO_FEATURE_NAME].shape, (8, 30720))


if __name__ == '__main__':
  tf.test.main()
