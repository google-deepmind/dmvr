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

"""Utils for adding modalities."""

import functools
from typing import Optional
from typing import Union

from absl import logging
from dmvr import builders
from dmvr import processors
from dmvr import tokenizers
import tensorflow as tf


# ----------------------------------------------------------------------
# -------- Methods aggregating functions for a given modality. ---------
# ----------------------------------------------------------------------


def add_image(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str = 'image/encoded',
    output_feature_name: str = builders.IMAGE_FEATURE_NAME,
    is_training: bool = True,
    # Video related parameters.
    num_frames: int = 32,
    stride: int = 1,
    num_test_clips: int = 1,
    min_resize: int = 224,
    resize_method: str = tf.image.ResizeMethod.BILINEAR,
    crop_size: int = 200,
    zero_centering_image: bool = False,
    sync_random_state: bool = True,
    is_rgb: Optional[bool] = True,
    is_flow: bool = False,
    random_flip: bool = True,
    normalization_mean: Union[tf.Tensor, float] = 0,
    normalization_std: Union[tf.Tensor, float] = 1,
) -> None:
  """Adds functions to process image feature to builders.

  This function expects the input to be either a `tf.train.SequenceExample` (for
  videos) and have the following structure:
  ```
  feature_lists {
    feature_list {
      key: input_feature_name
      value {
        feature {
          bytes_list {
            value: jpeg_bytes
          }
        }
      }
    }
  }
  ```

  Or a `tf.train.Example` (for image only) and have the following structure:
  ```
  features {
    feature {
      key: input_feature_name
      value {
        bytes_list {
          value: "JPEG"
        }
      }
    }
  }
  ```

  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    input_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different image features within a single dataset.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different
      image features within a single dataset.
    is_training: Whether in training mode. If `True`, random sample, crop and
      left right flip is used.
    num_frames: Number of frames per subclip. For single images, use 1.
    stride: Temporal stride to sample frames.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each video at test time.
      If 1, then a single clip in the middle of the video is sampled. The clips
      are aggregated in the batch dimension.
    min_resize: Frames are resized so that `min(height, width)` is `min_resize`.
    resize_method: A resizing method.
    crop_size: Final size of the frame after cropping the resized frames. Both
      height and width are the same.
    zero_centering_image: If `True`, frames are normalized to values in [-1, 1].
      If `False`, values in [0, 1].
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations such as sampling and
      cropping.
    is_rgb: If `True`, the number of channels in the JPEG is 3, if False, 1. If
      is_flow is `True`, `is_rgb` should be set to `None` (see below).
    is_flow: If `True`, the image is assumed to contain flow and will be
      processed as such. Note that the number of channels in the JPEG for flow
      is 3, but only two channels will be output corresponding to the valid
      horizontal and vertical displacement.
    random_flip: If `True`, a random horizontal flip is applied to the input
      image. This augmentation may not be used if the label set contains
      direction related classes, such as `pointing left`, `pointing right`, etc.
    normalization_mean: value to subtract from the input image to normalize it.
    normalization_std: value to divide by from the input image to normalize it.
  """

  # Validate parameters.
  if is_flow and is_rgb is not None:
    raise ValueError('`is_rgb` should be `None` when requesting flow.')

  if is_flow and not zero_centering_image:
    raise ValueError('Flow contains displacement values that can be negative, '
                     'but `zero_centering_image` was set to `False`.')

  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)

  # Parse frames or single image.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenSequenceFeature((), dtype=tf.string),
        output_name=output_feature_name)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.FixedLenFeature((), dtype=tf.string),
        output_name=output_feature_name)
    # Expand dimensions so single images have the same structure as videos.
    sampler_builder.add_fn(
        fn=lambda x: tf.expand_dims(x, axis=0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_expand_dims')
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Temporal sampler.
  if is_training:
    # Sample random clip.
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.sample_sequence(
            x, num_frames, True, stride, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    if num_test_clips > 1:
      # Sample linspace clips.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: processors.sample_linspace_sequence(
              x, num_test_clips, num_frames, stride),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_linspace_sample')
    else:
      # Sample middle clip.
      sampler_builder.add_fn(
          fn=lambda x: processors.sample_sequence(x, num_frames, False, stride),
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_middle_sample')

  # Decode JPEG string to `tf.uint8`.
  # Note that for flow, 3 channels are stored in the JPEG: the first two
  # corresponds to horizontal and vertical displacement, respectively.
  # The last channel contains zeros and is dropped later in the preprocessing.
  # Hence, the output number of channels for flow is 2.
  num_raw_channels = 3 if (is_rgb or is_flow) else 1
  decoder_builder.add_fn(
      fn=lambda x: processors.decode_jpeg(x, channels=num_raw_channels),
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_decode_jpeg')

  if is_flow:
    # Cast the flow to `tf.float32`, normalizing between [-1.0, 1.0].
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image=True),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize')

  # Resize images (resize happens only if necessary to save compute).
  preprocessor_builder.add_fn(
      # pylint: disable=g-long-lambda
      fn=lambda x: processors.resize_smallest(
          x, min_resize, is_flow=is_flow, method=resize_method),
      # pylint: enable=g-long-lambda
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_resize_smallest')

  if is_training:
    # Standard image data augmentation: random crop and random flip.
    preprocessor_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.crop_image(
            x, crop_size, crop_size, True, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_crop',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
    if random_flip:
      preprocessor_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x, s=None: processors.random_flip_left_right(
              x, state=s, is_flow=is_flow),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_random_flip',
          # Use state to keep coherence between modalities if requested.
          stateful=sync_random_state)
  else:
    # Central crop of the frames.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.crop_image(x, crop_size, crop_size, False),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_central_crop')

  if is_flow:
    # Keep only two channels for the flow: horizontal and vertical displacement.
    preprocessor_builder.add_fn(
        fn=lambda x: x[:, :, :, :2],
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_extract_flow_channels')

    # Clip the flow to stay between [-1.0 and 1.0]
    preprocessor_builder.add_fn(
        fn=lambda x: tf.clip_by_value(x, -1.0, 1.0),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_clip_flow')
  else:
    # Cast the frames to `tf.float32`, normalizing according to
    # `zero_centering_image`.
    preprocessor_builder.add_fn(
        fn=lambda x: processors.normalize_image(x, zero_centering_image),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_normalize')

  preprocessor_builder.add_fn(
      fn=lambda x: x - normalization_mean,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_subtract_given_mean')

  preprocessor_builder.add_fn(
      fn=lambda x: x / normalization_std,
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_divide_by_given_std')

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimension which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(  # pylint: disable=g-long-lambda
            x, (-1, num_frames, x.shape[2], x.shape[3], x.shape[4])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')


def add_label(
    parser_builder: builders.BaseParserBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    input_label_index_feature_name: str = 'clip/label/index',
    output_label_index_feature_name: str = builders.LABEL_INDEX_FEATURE_NAME,
    input_label_name_feature_name: Optional[str] = 'clip/label/text',
    output_label_name_feature_name: Optional[str] = builders
    .LABEL_NAME_FEATURE_NAME,
    # Label related parameters.
    is_multi_label: bool = False,
    one_hot_label: bool = True,
    num_classes: Optional[int] = None,
    add_label_name: bool = False):
  """Adds functions to process label feature to builders.

  This function expects the input to be either a `tf.train.SequenceExample`
  (with the features in the context) or a `tf.train.Example`. The expected
  structure is (or equivalent for `tf.train.Example`):
  ```
  context {
    feature {
      key: input_label_index_feature_name
      value {
        int64_list {
          value: 42
          ...
        }
      }
    }
    feature {
      key: input_label_name_feature_name
      value {
        bytes_list {
          value: "label_42"
          ...
        }
      }
    }
  }
  ```

  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    input_label_index_feature_name: Name of the label index feature in the input
      `tf.train.Example` or `tf.train.SequenceExample`. Exposing this as an
      argument allows using this function for different label features within a
      single dataset.
    output_label_index_feature_name: Name of the label index feature in the
      output features dictionary. Exposing this as an argument allows using this
      function for different label features within a single dataset.
    input_label_name_feature_name: Name of the label name feature in the input
      `tf.train.Example` or `tf.train.SequenceExample`. If `add_label_name` is
      false, this option is ignored. Exposing this as an argument allows using
      this function for different label features within a single dataset.
    output_label_name_feature_name: Name of the label name feature in the output
      features dictionary. If `add_label_name` is false, this option is ignored.
      Exposing this as an argument allows using this function for different
      label features within a single dataset.
    is_multi_label: Whether raw data contains multiple labels per example.
    one_hot_label: Return labels as one hot tensors. If `is_multi_label` is
      `True`, one hot tensor might have multiple ones.
    num_classes: Total number of classes in the dataset. It has to be provided
      if `one_hot_label` is `True`.
    add_label_name: Also return the name of the label. Not yet supported for
      multi label.
  """
  # Validate parameters.
  if one_hot_label and not num_classes:
    raise ValueError(
        '`num_classes` must be given when requesting one hot label.')
  if is_multi_label and not one_hot_label:
    logging.warning(
        'Multi label indices will be returned in a non fixed size dimension.')
  if add_label_name and (input_label_name_feature_name is None or
                         output_label_name_feature_name is None):
    raise ValueError(
        '`input_label_name_feature_name` and `output_label_name_feature_name` '
        'must be given when `add_label_name` is true.')

  # Parse label.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_label_index_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.int64),
        output_name=output_label_index_feature_name,
        is_context=True)
    if add_label_name:
      parser_builder.parse_feature(
          feature_name=input_label_name_feature_name,
          feature_type=tf.io.VarLenFeature(dtype=tf.string),
          output_name=output_label_name_feature_name,
          is_context=True)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_label_index_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.int64),
        output_name=output_label_index_feature_name)
    if add_label_name:
      parser_builder.parse_feature(
          feature_name=input_label_name_feature_name,
          feature_type=tf.io.VarLenFeature(dtype=tf.string),
          output_name=output_label_name_feature_name)
  else:
    raise ValueError('`parser_builder` has an unexpected type.')

  # Densify labels tensor in order to support multi label case.
  decoder_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_label_index_feature_name,
      fn_name=f'{output_label_index_feature_name}_sparse_to_dense')
  if add_label_name:
    decoder_builder.add_fn(
        fn=tf.sparse.to_dense,
        feature_name=output_label_name_feature_name,
        fn_name=f'{output_label_name_feature_name}_sparse_to_dense')

  if one_hot_label:
    # Replace label index by one hot representation.
    preprocessor_builder.add_fn(
        fn=lambda x: tf.reduce_sum(  # pylint: disable=g-long-lambda
            input_tensor=tf.one_hot(x, num_classes),
            axis=0),
        feature_name=output_label_index_feature_name,
        fn_name=f'{output_label_index_feature_name}_one_hot')
  elif not is_multi_label:
    preprocessor_builder.add_fn(
        fn=lambda x: processors.set_shape(x, (1,)),
        feature_name=output_label_index_feature_name,
        fn_name=f'{output_label_index_feature_name}_set_shape')

  if add_label_name and not is_multi_label:
    preprocessor_builder.add_fn(
        fn=lambda x: processors.set_shape(x, (1,)),
        feature_name=output_label_name_feature_name,
        fn_name=f'{output_label_name_feature_name}_set_shape')


def add_text(
    parser_builder: builders.BaseParserBuilder,
    decoder_builder: builders.DecoderBuilder,
    preprocessor_builder: builders.PreprocessorBuilder,
    tokenizer: tokenizers.TextTokenizer,
    is_training: bool = True,
    input_feature_name: str = 'caption/string',
    output_raw_string_name: str = builders.TEXT_FEATURE_NAME,
    output_feature_name: str = builders.TEXT_INDICES_FEATURE_NAME,
    # Text related parameters.
    prepend_bos: bool = False,
    append_eos: bool = False,
    keep_raw_string: bool = False,
    max_num_captions: int = 1,
    max_num_tokens: Optional[int] = 16,
    sync_random_state: bool = False):
  """Adds functions to process text feature to builders.

  This function expects the input to be either a `tf.train.SequenceExample`
  (with the features in the context) or a `tf.train.Example`. The expected
  structure is (or equivalent for `tf.train.Example`):
  ```
  context {
    feature {
      key: input_feature_name
      value {
        bytes_list {
          value: "Hello world!"
          value: "This is a caption."
          ...
        }
      }
    }
  }
  ```

  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    decoder_builder: An instance of a `builders.DecoderBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    tokenizer: An instance of a tokenizer.
    is_training: Whether in training mode. This will be used to randomly sample
      the captions.
    input_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different text features within a single dataset.
    output_raw_string_name: Name of the raw string in the output features
      dictionary. Exposing this as an argument allows using this function for
      different text features within a single dataset.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different text
      features.
    prepend_bos: Whether to prepend BOS token.
    append_eos: Whether to append EOS token.
    keep_raw_string: Whether to keep raw string.
    max_num_captions: Maximum number of captions to keep. If there are more
      captions in the proto, only the first `max_num_captions` will be returned
      is `is_training` is set to `False`. If `is_training` is `True`, then
      `max_num_captions` will be randomly sampled. Finally, if the proto
      contains less than `max_num_captions`, we pad with empty strings to make
      sure there are `max_num_captions` in total.
    max_num_tokens: Maximum number of tokens to keep from the text for each
      caption. If there are more tokens, sequence is cropped, if less, the
      caption is padded using the tokenizer pad id. The sequence is unmodified
      if max_num_tokens is None.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations used for sampling
      the captions.
  """
  # Parse text indices.
  if isinstance(parser_builder, builders.SequenceExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name,
        is_context=True)
  elif isinstance(parser_builder, builders.ExampleParserBuilder):
    parser_builder.parse_feature(
        feature_name=input_feature_name,
        feature_type=tf.io.VarLenFeature(dtype=tf.string),
        output_name=output_raw_string_name)

  # Densify text tensor.
  decoder_builder.add_fn(
      fn=tf.sparse.to_dense,
      feature_name=output_raw_string_name,
      fn_name=f'{output_feature_name}_sparse_to_dense')

  preprocessor_builder.add_fn(
      # pylint: disable=g-long-lambda
      lambda x, s=None: processors.sample_or_pad_non_sorted_sequence(
          x, max_num_captions, b'', random=is_training, state=s),
      # pylint: enable=g-long-lambda
      feature_name=output_raw_string_name,
      fn_name=f'{output_feature_name}_sample_captions',
      # Use state to keep coherence between modalities if requested.
      stateful=sync_random_state)

  # Tokenize the sentence.
  preprocessor_builder.add_fn(
      fn=lambda x: processors.tokenize(  # pylint: disable=g-long-lambda
          x, tokenizer, output_raw_string_name, output_feature_name,
          prepend_bos, append_eos, max_num_tokens, keep_raw_string),
      fn_name=f'{output_feature_name}_tokenization')

  if max_num_tokens is not None:
    # Set text shape.
    shape = (max_num_captions, max_num_tokens)
    preprocessor_builder.add_fn(
        fn=lambda x: processors.set_shape(x, shape),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_set_shape')


def add_audio(
    parser_builder: builders.BaseParserBuilder,
    sampler_builder: builders.SamplerBuilder,
    postprocessor_builder: builders.PostprocessorBuilder,
    preprocessor_builder: Optional[builders.PreprocessorBuilder] = None,
    input_feature_name: str = 'WAVEFORM/feature/floats',
    output_feature_name: str = builders.AUDIO_FEATURE_NAME,
    is_training: bool = True,
    # Audio related parameters.
    num_samples: int = 30720,
    stride: int = 1,
    sample_rate: Optional[int] = 48000,
    target_sample_rate: Optional[int] = None,
    num_test_clips: int = 1,
    sync_random_state: bool = True):
  """Adds functions to process audio feature to builders.

  This function expects the input to be either a `tf.train.SequenceExample` (for
  videos) and have the following structure:
  ```
  feature_lists {
    feature_list {
      key: input_feature_name
      value {
        feature {
          float_list {
            value: 0.0
            value: 0.1
            value: 0.2
            ...
          }
        }
      }
    }
  }
  ```

  Or a `tf.train.Example` (for image only) and have the following structure:
  ```
  features {
    feature {
      key: input_feature_name
      value {
        float_list {
          value: 0.0
          value: 0.1
          value: 0.2
          ...
        }
      }
    }
  }
  ```

  The corresponding `builders.ExampleParserBuilder` or
  `builders.SequenceExampleParserBuilder` has to be given as parameter.

  Args:
    parser_builder: An instance of a `builders.BaseParserBuilder`.
    sampler_builder: An instance of a `builders.SamplerBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    input_feature_name: Name of the feature in the input `tf.train.Example` or
      `tf.train.SequenceExample`. Exposing this as an argument allows using this
      function for different audio features within a single dataset.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different
      audio features within a single dataset
    is_training: Whether in training mode. If `True`, random sample is used.
    num_samples: Number of samples per subclip.
    stride: Temporal stride to sample audio signal.
    sample_rate: The original sample rate of the input audio stored in sstables.
    target_sample_rate: If this is not None, the target new sample rate of the
      waveforms. Fast Fourier Transforms will be triggered if true.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggregated in the batch dimension.
    sync_random_state: Whether to use stateful option to keep random operations
      in sync between different modalities. All modalities having this option
      `True` will use the same outcome in random operations such as sampling and
      cropping.
  """
  # Validate parameters.
  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)

  # Keep audio signal.
  parser_builder.parse_feature(
      feature_name=input_feature_name,
      # Entire signal stored in one Feature.
      feature_type=tf.io.VarLenFeature(dtype=tf.float32),
      output_name=output_feature_name)

  # Densify.
  sampler_builder.add_fn(
      fn=lambda x: tf.sparse.to_dense(x)[0],
      feature_name=output_feature_name,
      fn_name=f'{output_feature_name}_sparse_to_dense')

  # Temporal sampler.
  if is_training:
    # Sample random clip.
    sampler_builder.add_fn(
        # pylint: disable=g-long-lambda
        fn=lambda x, s=None: processors.sample_sequence(
            x, num_samples, True, stride, state=s),
        # pylint: enable=g-long-lambda
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_random_sample',
        # Use state to keep coherence between modalities if requested.
        stateful=sync_random_state)
  else:
    if num_test_clips > 1:
      # Sample linspace clips.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: processors.sample_linspace_sequence(
              x, num_test_clips, num_samples, stride),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_linspace_sample')
    else:
      # Sample middle clip.
      sampler_builder.add_fn(
          # pylint: disable=g-long-lambda
          fn=lambda x: processors.sample_sequence(
              x, num_samples, False, stride),
          # pylint: enable=g-long-lambda
          feature_name=output_feature_name,
          fn_name=f'{output_feature_name}_middle_sample')

  # Apply FFTs to change the sample rate of the waveforms.
  if preprocessor_builder is not None and target_sample_rate is not None:
    preprocessor_builder.add_fn(
        functools.partial(
            processors.resample_audio,
            num_subclips=num_test_clips,
            in_sample_rate=sample_rate,
            out_sample_rate=target_sample_rate,
            is_training=is_training),
        feature_name=builders.AUDIO_FEATURE_NAME)

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimension which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(x, (-1, x.shape[-1])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')


def add_spectrogram(
    preprocessor_builder: builders.PreprocessorBuilder,
    postprocessor_builder: builders.PostprocessorBuilder,
    input_feature_name: str = builders.AUDIO_FEATURE_NAME,
    output_feature_name: str = builders.AUDIO_MEL_FEATURE_NAME,
    is_training: bool = True,
    sample_rate: int = 48000,
    spectrogram_type: str = 'logmf',
    frame_length: int = 2048,
    frame_step: int = 1024,
    num_features: int = 80,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    preemphasis: Optional[float] = None,
    normalize_audio: bool = False,
    num_test_clips: int = 1):
  """Adds functions to process audio spectrogram feature to builders.

  Note that this function does not extract and parse audio feature. Instead, it
  should be used after a `add_audio` function. The output spectrogram is of the
  shape [batch_size, num_frames, num_features].

  Args:
    preprocessor_builder: An instance of a `builders.PreprocessorBuilder`.
    postprocessor_builder: An instance of a `builders.PostprocessorBuilder`.
    input_feature_name: Name of the feature in the input features dictionary.
      Exposing this as an argument allows using this function for different
      audio features.
    output_feature_name: Name of the feature in the output features dictionary.
      Exposing this as an argument allows using this function for different
      audio features.
    is_training: If the current mode is training or not.
    sample_rate: The sample rate of the input audio.
    spectrogram_type: The type of the spectrogram to be extracted from the
      waveform. Can be either `spectrogram`, `logmf`, and `mfcc`.
    frame_length: The length of each spectrogram frame.
    frame_step: The stride of spectrogram frames.
    num_features: The number of spectrogram features.
    lower_edge_hertz: Lowest frequency to consider.
    upper_edge_hertz: Highest frequency to consider.
    preemphasis: The strength of pre-emphasis on the waveform. If None, no
      pre-emphasis will be applied.
    normalize_audio: Whether to normalize the waveform or not.
    num_test_clips: Number of test clips (1 by default). If more than 1, this
      will sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggregated in the batch dimension.
  """
  # Validate parameters.
  if is_training and num_test_clips != 1:
    logging.info('`num_test_clips` %d is ignored since `is_training` is true.',
                 num_test_clips)

  # Extract audio spectrograms.
  preprocessor_builder.add_fn(
      functools.partial(
          processors.compute_audio_spectrogram,
          num_subclips=num_test_clips,
          sample_rate=sample_rate,
          spectrogram_type=spectrogram_type,
          frame_length=frame_length,
          frame_step=frame_step,
          num_features=num_features,
          lower_edge_hertz=lower_edge_hertz,
          upper_edge_hertz=upper_edge_hertz,
          normalize=normalize_audio,
          preemphasis=preemphasis,
          audio_feature_name=input_feature_name,
          spectrogram_feature_name=output_feature_name))

  if num_test_clips > 1 and not is_training:
    # In this case, multiple clips are merged together in batch dimension which
    # will be `B * num_test_clips`.
    postprocessor_builder.add_fn(
        fn=lambda x: tf.reshape(x, (-1, x.shape[-2], x.shape[-1])),
        feature_name=output_feature_name,
        fn_name=f'{output_feature_name}_reshape')
