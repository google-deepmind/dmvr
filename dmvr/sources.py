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

"""Sources for reading and decoding raw binary data files."""

import abc
from typing import Optional, Union

import tensorflow as tf


class Source(abc.ABC):
  """Base class for sources.

  Sources are objects reading from binary files and generating an initial
  `tf.data.Dataset` with the serialized examples. Deserializing the examples is
  not responsibility of the `Source` (it should be done by the parser).

  For each different type of storage (e.g. TFRecords, image files, text files),
  a subclass can be implemented.
  """

  @abc.abstractmethod
  def load_and_decode_shard(
      self,
      shard: Union[str, tf.Tensor]  # Shape () and type `tf.string`.
  ) -> tf.data.Dataset:
    """Decodes a single raw input file into a `tf.data.Dataset`.

    Args:
      shard: Path to a single file with encoded data.

    Returns:
      A `tf.data.Dataset` object containing a key (this can be a file name,
      index, empty or any other useful bits) and a raw example (both encoded as
      bytes). Current supported types of examples are `tf.train.Example` and
      `tf.train.SequenceExample` (see `builders.BaseParserBuilder`).
    """


class TFRecordsSource(Source):
  """Source for TFRecords data format."""

  def __init__(self, compression_type: Optional[str] = None):
    self._compression_type = compression_type

  def load_and_decode_shard(
      self,
      shard: Union[str, tf.Tensor]  # Shape () and type `tf.string`.
  ) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(shard, compression_type=self._compression_type)
    # TFRecords do not provide an index or key per example. Use shard path as
    # key, since it can be useful later for retrieval.
    key = shard.encode('utf-8') if isinstance(shard, str) else shard
    ds = ds.map(lambda example: (key, example))
    return ds
