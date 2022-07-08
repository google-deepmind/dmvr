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
"""Basic constructors for video datasets."""

import abc
from typing import Any, List, Optional, Type, TypeVar

from absl import logging
from dmvr import builders
from dmvr import sources
import tensorflow as tf

# Types.
T = TypeVar('T', bound=builders.BaseParserBuilder)
NestedStructure = Any


class BaseVideoDatasetFactory(abc.ABC):
  """Base class to build final `tf.data.Dataset` objects from files.

  Glossary:

  - A source is an object reading binary files in disk (e.g. TFRecords, image
  files) and outputting serialized examples (e.g. `tf.train.SequenceExample`).
  - A parser is an object reading serialized examples (e.g.
  `tf.train.SequenceExample`) and outputting a `builders.FeaturesDict`.
  - A processor is an object transforming features dictionary.
  - The data processing pipeline is organised in phases. A phase is an unit of
  the data processing graph and will have one parser or processor.
  - Builders are helpers designed to allow the user to easily customize the data
  processing graph by adding functions to each phase.

  Principle:

  All datasets created with this factory follow the same abstraction:
  a `parse_fn`, a `sample_fn`, a `decode_fn`, a `preprocess_fn` and a
  `postprocess_fn` are used to control the flow of dataset creation besides
  normal dataset operations. These functions are created from builders, allowing
  the user to build a graph of data processing operations. In details, the
  following steps are followed when creating a dataset:
    - Read shards from file system using the given `source.Source`.
    - Apply `parse_fn` to output values of the `source` (as bytes) to build a
    dictionary of raw features. The parse function should only parse the useful
    bytes of the serialized input example (e.g. `tf.train.SequenceExample`) and
    put the features in a `builders.FeaturesDict` format. `parser_builder` can
    be used to easily add more features / modalities.
    - Apply `sample_fn` to sequence features contained in the dictionary in
    order to select the desired elements of the sequence, e.g. sample a subset
    of frames from the entire stored video. `sampler_builder` can be used to
    modify or add sampling options.
    - Apply `decode_fn` to convert raw formats to the final format. E.g. decode
    JPEG string `tf.Tensor` to a `tf.Tensor` of `uint8`. `decoder_builder` can
    be used.
    - Apply `preprocess_fn`. E.g. crop images, process audio and text.
    `preprocessor_builder` can be used.
    - Batch, shuffle, prefetch and do other basic operations with the dataset.
    - Apply `postprocess_fn` to batched examples. E.g. transpose batches.
    `postprocessor_builder` can be used.

  After each one of the data processing functions, a filter is applied in order
  to keep only desirable elements in the dataset. These filters can be
  customized by using the `filter_builder`.

  A conventional use of this factory consists of implementing a subclass for a
  specific dataset, overriding the `_build` method where all common processing
  of the specific dataset can be added using the builders.

  The client of the dataset is able to create a factory, configure it, possibly
  add custom extra processing steps and use it to make a dataset.

  Usage:

  ```python
  class KineticsFactory(BaseVideoDatasetFactory):

    def __init__(self, subset: str):
      shards = ['path/to/kinetics/tfrecords/records-00001-of-00500.tfrecord',
                ...]
      shards = filter_by_subset(shards, subset)
      super().__init__(shards)

    def _build(self, frame_height: int, frame_width: int, frame_count: int):
      self.parser_builder.parse_feature(
          image_seq_example_feature_name,
          tf.io.FixedLenSequenceFeature((), dtype=tf.string),
          builders.IMAGE_FEATURE_NAME)
      self.sampler_builder.add_fn(
          lambda x: sample_sequence_fn(x, frame_count),
          builders.IMAGE_FEATURE_NAME)
      self.decoder_builder.add_fn(decode_frames_fn, builders.IMAGE_FEATURE_NAME)
      self.preprocessor_builder.add_fn(
          lambda x: resize_frames(x, frame_height, frame_width),
          builders.IMAGE_FEATURE_NAME)
      # Other processing functions adding text and label.

  # Dataset client code:
  factory = KineticsFactory(subset='test').configure(
      frame_height=224, frame_width=224, frame_count=8)

  # Add extra custom preprocess functions:
  def my_custom_text_tokenizer(text: tf.Tensor) -> tf.Tensor:
    # Tokenize text string.
    return tokenized_tensor

  def my_custom_add_word_indices(
      features_dict: builders.FeaturesDict) -> builders.FeaturesDict:
    tokenized_text = features_dict[builders.TEXT_FEATURE_NAME]
    features_dict[builders.TEXT_INDICES_FEATURE_NAME] = text_to_indices(
        tokenized_text)
    return features_dict

  (factory.preprocess_builder.add_fn(my_custom_tokenizer,
                                     builders.TEXT_FEATURE_NAME)
                             .add_fn(my_custom_add_word_indices))

  # Add filter:
  def keep_only_label_zero(fetures_dict: builders.FeaturesDict) -> tf.Tensor:
    return tf.equal(features_dict[builders.LABEL_INDEX_FEATURE_NAME], 0)
  factory.filter_builder.add_filter_fn(
      keep_only_label_zero, builders.Phase.PARSE)

  # Create dataset:
  ds = factory.make_dataset(batch_size=16)
  ```

  The factory exposes the process functions builders to the client, allowing
  simple modifications to the functions. Common process functions, as crop,
  resize, etc. should be implemented in common modules.

  See builders documentation for more details.
  """

  def __init__(self,
               shards: List[str],
               parser_builder_class: Type[T] = builders
               .SequenceExampleParserBuilder,
               source: sources.Source = sources.TFRecordsSource()):
    """Initializes the `BaseVideoDatasetFactory`.

    Args:
      shards: List of paths to shards containing the data files. Each one of the
        paths will be passed to the `source`, that will read the data and output
        examples (that will be fed into the parse function generated by the
        `parser_builder_class`). Therefore, `shards`, `parser_builder_class` and
        `source` have to be consistent.
      parser_builder_class: A parser builder class able to parse examples of the
        types contained in `shards` files.
      source: Source to be used to load raw binary files and decoding it into
        examples (encoded as bytes).
    """

    self._shards = shards
    self._source = source

    # Initialize all function builders.
    self.parser_builder = parser_builder_class()
    self.sampler_builder = builders.SamplerBuilder()
    self.decoder_builder = builders.DecoderBuilder()
    self.preprocessor_builder = builders.PreprocessorBuilder()
    self.postprocessor_builder = builders.PostprocessorBuilder()

    # Initialize filters.
    self.filter_builder = builders.FilterBuilder()

    # Default tune parameters.
    self._shuffle_buffer = 256
    self._num_parser_threads = 16
    self._num_process_threads = tf.data.experimental.AUTOTUNE
    self._num_postprocess_threads = 4
    self._parser_buffer_size = 64
    self._postprocess_buffer_size = 1
    self._prefetch_buffer_size = 8
    self._cycle_length = None
    self._num_parallel_calls_interleave = tf.data.experimental.AUTOTUNE
    self._block_length = None
    self._seed = None
    self._duplicate_proto = None

    self._is_configured = False

  def configure(self, *args, **kwargs) -> 'BaseVideoDatasetFactory':
    """Configures all parse and process functions of this factory.

    This function should be called exactly once per factory instance and will
    delegate builders configuration to `_build` method.

    Args:
      *args: Positional arguments passed to `_build` function.
      **kwargs: Non positional arguments passed to `_build` function.

    Returns:
      This instance of the factory.

    Raises:
      ValueError: Method has already been called.
    """
    if self._is_configured:
      raise ValueError(
          '`configure` has already been called. The method should be called '
          'only once to avoid duplicated process functions.')
    self._is_configured = True
    self._build(*args, **kwargs)
    return self

  def tune(self,
           shuffle_buffer: Optional[int] = None,
           num_parser_threads: Optional[int] = None,
           num_process_threads: Optional[int] = None,
           num_postprocess_threads: Optional[int] = None,
           parser_buffer_size: Optional[int] = None,
           postprocess_buffer_size: Optional[int] = None,
           prefetch_buffer_size: Optional[int] = None,
           cycle_length: Optional[int] = None,
           num_parallel_calls_interleave: Optional[int] = None,
           block_length: Optional[int] = None,
           seed: Optional[int] = None,
           duplicate_proto: Optional[int] = None):
    """Changes the dataset creation parameters.

    This method should be used to change the default parameters used to create
    the dataset in order to improve speed, memory or other. Only given
    parameters will be changed, the others will remain the same.

    Args:
      shuffle_buffer: The buffer size for shuffle operation. This affects the
        randomness of the output. It must be specified if `shuffle` is `True`.
      num_parser_threads: Number of threads to use for the parsing operation.
        `tf.data.experimental.AUTOTUNE` can be used to auto-tune.
      num_process_threads: Number of threads to use for map operations in
        sample, decode and preprocess. `tf.data.experimental.AUTOTUNE` can be
        used to auto-tune.
      num_postprocess_threads: Number of threads to use for map operations in
        postprocess. `tf.data.experimental.AUTOTUNE` can be used to auto-tune.
      parser_buffer_size: Buffer size of the sample, decode and preprocess
        operation.
      postprocess_buffer_size: Buffer size of the postprocess operation.
      prefetch_buffer_size: Size of the final prefetch buffer.
      cycle_length: The number of shards that will be processed concurrently.
        `tf.data.experimental.AUTOTUNE` can be used to auto-tune.
      num_parallel_calls_interleave: The number of parallel calls to the
        interleave method. `tf.data.experimental.AUTOTUNE` can be used to
        auto-tune.
      block_length: The number of consecutive elements to produce from each
        shard.
      seed: Random seed of the shuffle operations.
      duplicate_proto: Number of duplicates to make for each loaded proto.
        Typically different augmentations will be applied for each copy, so
        this can reduce disk reads without harming training performance.
        This is applied after the post read function, but before the shuffle
        buffer.

    Returns:
      This instance of the factory.
    """
    self._shuffle_buffer = shuffle_buffer or self._shuffle_buffer
    self._num_parser_threads = num_parser_threads or self._num_parser_threads
    self._num_process_threads = num_process_threads or self._num_process_threads
    self._num_postprocess_threads = (
        num_postprocess_threads or self._num_postprocess_threads)
    self._parser_buffer_size = parser_buffer_size or self._parser_buffer_size
    self._postprocess_buffer_size = (
        postprocess_buffer_size or self._postprocess_buffer_size)
    self._prefetch_buffer_size = (
        prefetch_buffer_size or self._prefetch_buffer_size)
    self._cycle_length = cycle_length or self._cycle_length
    self._num_parallel_calls_interleave = (
        num_parallel_calls_interleave or self._num_parallel_calls_interleave)
    self._block_length = block_length or self._block_length
    self._seed = seed or self._seed
    self._duplicate_proto = duplicate_proto or self._duplicate_proto

    return self

  # ----------------------------------------------------------------------
  # ---------- Methods that must be implemented by child class. ----------
  # ----------------------------------------------------------------------

  @abc.abstractmethod
  def _build(self, *args, **kwargs) -> None:
    """Builds the data processing graph."""

  # ----------------------------------------------------------------------
  # -------- Methods that should only be overridden if necessary. --------
  # ----------------------------------------------------------------------

  def make_dataset(
      self,
      shuffle: bool = True,
      num_epochs: Optional[int] = None,
      batch_size: Optional[int] = 16,
      padded_batch: bool = False,
      padded_batch_shapes: NestedStructure = None,
      drop_remainder: bool = True,
      keep_key: bool = False,
      cache: bool = False,
      override_preprocess_fn: Optional[builders.Processor] = None,
      **experimental_kwargs
  ) -> tf.data.Dataset:
    """Creates a `tf.data.Dataset` instance of the given dataset.

    Args:
      shuffle: Whether output data is shuffled.
      num_epochs: Number of epochs to cycle through before stopping. If `None`,
        this will read samples indefinitely.
      batch_size: If an int, an extra leading batch dimension will be present
        for all features. If `None`, then no batching is done and no extra batch
        dimension is added.
      padded_batch: Whether to use `padded_batch` instead of `batch` method.
        Padded batch pads a batch of examples to a given output shape. It pads
        all examples to the longest one in that batch. This could be used for
        sequence data.
      padded_batch_shapes: `padded_shapes` to be passed to `padded_batch`.
      drop_remainder: Whether to drop any remainder after the last full-size
        batch. If `True`, the batch dimension of the resulting op is known;
        otherwise, the batch dimension may be `None` in cases where `num_epochs`
        is finite and `batch_size` > 1, since the final remainder batch may be
        smaller than the usual batch size.
      keep_key: Whether to keep the `builders.Source` key as a feature in the
        final dictionary. The key for the key in the dictionary is
        `builders.KEY_FEATURE_NAME`.
      cache: Whether to cache the dataset in RAM. Note that this should only
        be used if the dataset can fit in RAM as otherwise it will lead to
        out of memory error.
      override_preprocess_fn: Function to use instead of built preprocess_fn.
      **experimental_kwargs: Other arguments used for experimental features.
        These can be removed at any time without prior notice.

    Returns:
      An instance of the dataset.

    Raises:
      ValueError: Factory has not been configured.
      ValueError: `shuffle_buffer` is `None` when dataset is shuffled.
      ValueError: `batch_size` is not `None`, `padded_batch` is `False` and
      `padded_batch_shapes` is not `None`.
    """

    if not self._is_configured:
      raise ValueError('Factory has not been configured. Call `configure` '
                       'method before `make_dataset`.')

    # Build functions or use its overrides.
    parse_fn = self.parser_builder.build()
    sample_fn = self.sampler_builder.build()
    decode_fn = self.decoder_builder.build()
    preprocess_fn = override_preprocess_fn or self.preprocessor_builder.build()
    postprocess_fn = self.postprocessor_builder.build()

    # Filter functions.
    filter_fn_post_read = self.filter_builder.build(builders.Phase.READ)
    filter_fn_post_parse = self.filter_builder.build(builders.Phase.PARSE)
    filter_fn_post_sample = self.filter_builder.build(builders.Phase.SAMPLE)
    filter_fn_post_decode = self.filter_builder.build(builders.Phase.DECODE)
    filter_fn_post_preprocess = self.filter_builder.build(
        builders.Phase.PREPROCESS)
    filter_fn_post_postprocess = self.filter_builder.build(
        builders.Phase.POSTPROCESS)

    if shuffle and self._shuffle_buffer is None:
      raise ValueError(
          '`shuffle_buffer` cannot be `None` if dataset is shuffled.')

    def parse_example(key: tf.Tensor,
                      raw_example: tf.Tensor) -> builders.FeaturesDict:
      """Decodes bytes of example and parse it into a features dictionary."""
      output = parse_fn(raw_example)
      # Potentially parse the key.
      if keep_key:
        output[builders.KEY_FEATURE_NAME] = key
      return output

    ds = tf.data.Dataset.from_tensor_slices(self._shards)
    if shuffle:
      # Shuffling the shards and not only the examples later is important.
      ds = ds.shuffle(len(self._shards), seed=self._seed)

    ds = ds.interleave(
        self._source.load_and_decode_shard,
        cycle_length=self._cycle_length,
        block_length=self._block_length,
        num_parallel_calls=self._num_parallel_calls_interleave,
        deterministic=not shuffle)

    # At this point, the features dictionary is not yet created. We artificially
    # create one with the key only to make the interface uniform.
    ds = ds.filter(
        lambda key, _: filter_fn_post_read({builders.KEY_FEATURE_NAME: key}))

    if self._duplicate_proto is not None:

      def duplicate_fn(x, y):
        return (tf.stack([x] * self._duplicate_proto),
                tf.stack([y] * self._duplicate_proto))

      ds = ds.map(duplicate_fn)
      ds = ds.unbatch()

    if not cache:
      ds = ds.repeat(num_epochs)
      if shuffle:
        ds = ds.shuffle(self._shuffle_buffer, seed=self._seed)

    # Parse.
    ds = ds.map(
        parse_example,
        num_parallel_calls=self._num_parser_threads,
        deterministic=not shuffle)
    ds = ds.filter(filter_fn_post_parse)

    if cache:
      # We cache the dataset after the parsing operation. This means that we
      # cache the raw protos before any random operations happen. This can avoid
      # IO issues when the dataset fits in RAM. Note that this is the optimal
      # place to cache the data (caching before would have no effect as that
      # would only be caching a list of files, caching after would be not
      # possible due to the random operations that needs to happen after the
      # `ds.repeat` operation, making it impossible to cache as the dataset
      # would be unbounded).
      ds = ds.cache()
      ds = ds.repeat(num_epochs)
      if shuffle:
        ds = ds.shuffle(self._shuffle_buffer, seed=self._seed)
    else:
      ds = ds.prefetch(self._parser_buffer_size)

    # Sample.
    ds = ds.map(
        sample_fn,
        num_parallel_calls=self._num_process_threads,
        deterministic=not shuffle)
    ds = ds.filter(filter_fn_post_sample)

    # Decode.
    ds = ds.map(
        decode_fn,
        num_parallel_calls=self._num_process_threads,
        deterministic=not shuffle)
    ds = ds.filter(filter_fn_post_decode)

    # Preprocess.
    ds = ds.map(
        preprocess_fn,
        num_parallel_calls=self._num_process_threads,
        deterministic=not shuffle)
    ds = ds.filter(filter_fn_post_preprocess)

    if experimental_kwargs.get('unbatch_after_preprocessing', False):
      ds = ds.unbatch()

    if experimental_kwargs.get('ignore_processing_errors', False):
      ds = ds.apply(tf.data.experimental.ignore_errors())

    if batch_size is not None:
      if padded_batch:
        ds = ds.padded_batch(
            batch_size=batch_size,
            padded_shapes=padded_batch_shapes,
            drop_remainder=drop_remainder)
      else:
        if padded_batch_shapes is not None:
          raise ValueError(
              '`padded_batch` is `False`, `padded_batch_shapes` must be `None`,'
              f'but is {padded_batch_shapes}.')
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    # Postprocess.
    ds = ds.prefetch(self._postprocess_buffer_size)
    ds = ds.map(
        postprocess_fn,
        num_parallel_calls=self._num_postprocess_threads,
        deterministic=not shuffle)
    ds = ds.filter(filter_fn_post_postprocess)

    ds = ds.prefetch(self._prefetch_buffer_size)

    logging.info('Dataset created successfully')

    return ds
