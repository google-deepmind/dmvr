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

"""Utils."""

from typing import Optional, Sequence

import tensorflow as tf


# ----------------------------------------------------------------------
# ------------------------ Experimental utils. -------------------------
# ----------------------------------------------------------------------


def combine_datasets(datasets: Sequence[tf.data.Dataset],
                     batch_size: int = 1,
                     weights: Optional[Sequence[float]] = None,
                     seed: Optional[int] = None) -> tf.data.Dataset:
  """Combines multiple datasets into a single one.

  THIS IS AN EXPERIMENTAL FEATURE AND MIGHT BE REMOVED AT ANY TIME.

  This function combines multiple datasets into a single one by sampling
  elements from each one with the given probabilities. All input datasets must
  have the same structure and Tensor shapes.

  Args:
    datasets: A list of batched datasets. All datasets should have the same
      structure and Tensor shapes.
    batch_size: Batch size of the resulting dataset.
    weights: A list of the same length as datasets of floats where `weights[i]`
      represents the probability with which an element should be sampled from
      `datasets[i]`. If `None`, defaults to a uniform distribution across
      datasets.
    seed: A deterministic seed to use when sampling.

  Returns:
    A dataset that interleaves elements from datasets at random, according to
    weights if provided, otherwise with uniform probability. The resulting
    dataset is batched.
  """
  datasets = [ds.unbatch() for ds in datasets]
  combined_ds = tf.data.experimental.sample_from_datasets(
      datasets, weights, seed)
  return combined_ds.batch(batch_size, drop_remainder=True)
