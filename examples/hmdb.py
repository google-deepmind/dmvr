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
"""HMDB51 video dataset."""

import os
from typing import Optional

from dmvr import modalities
from dmvr import video_dataset


class HMDB51Factory(video_dataset.BaseVideoDatasetFactory):
  """HMDB51 reader."""

  _SUBSETS = ('train', 'test')
  _SPLITS = (1, 2, 3)
  _NUM_CLASSES = 51

  _NUM_SHARDS = {'train': 59, 'test': 39}

  def __init__(
      self,
      base_dir: str,
      subset: str = 'train',
      split: int = 1):
    """Constructor of HMDB51Factory."""

    if subset not in HMDB51Factory._SUBSETS:
      raise ValueError('Invalid subset "{}". The available subsets are: {}'
                       .format(subset, HMDB51Factory._SUBSETS))

    if split not in HMDB51Factory._SPLITS:
      raise ValueError('Invalid split "{}". The available splits are: {}'
                       .format(split, HMDB51Factory._SPLITS))

    num_shards = self._NUM_SHARDS[subset]
    shards = [f'{subset}_{split}-{i:05d}-of-{num_shards:05d}'
              for i in range(num_shards)]
    super().__init__(shards=[os.path.join(base_dir, s) for s in shards])

  def _build(self,
             is_training: Optional[bool] = True,
             # Video related parameters.
             num_frames: int = 32,
             stride: int = 1,
             num_test_clips: int = 1,
             min_resize: int = 256,
             crop_size: int = 224,
             zero_centering_image: bool = False,
             # Label related parameters.
             one_hot_label: bool = True,
             add_label_name: bool = False):
    """Default build for this dataset.

    Args:
      is_training: Whether or not in training mode.
      num_frames: Number of frames per subclip. For single images, use 1.
      stride: Temporal stride to sample frames.
      num_test_clips: Number of test clips (1 by default). If more than 1, this
        will sample multiple linearly spaced clips within each video at test
        time. If 1, then a single clip in the middle of the video is sampled.
        The clips are aggreagated in the batch dimension.
      min_resize: Frames are resized so that `min(height, width)` is
        `min_resize`.
      crop_size: Final size of the frame after cropping the resized frames. Both
        height and width are the same.
      zero_centering_image: If `True`, frames are normalized to values in
        [-1, 1]. If `False`, values in [0, 1].
      one_hot_label: Return labels as one hot tensors.
      add_label_name: Also return the name of the label.
    """
    modalities.add_image(
        parser_builder=self.parser_builder,
        sampler_builder=self.sampler_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        postprocessor_builder=self.postprocessor_builder,
        is_training=is_training,
        num_frames=num_frames, stride=stride,
        num_test_clips=num_test_clips,
        min_resize=min_resize, crop_size=crop_size,
        zero_centering_image=zero_centering_image)

    modalities.add_label(
        parser_builder=self.parser_builder,
        decoder_builder=self.decoder_builder,
        preprocessor_builder=self.preprocessor_builder,
        one_hot_label=one_hot_label,
        num_classes=HMDB51Factory._NUM_CLASSES,
        add_label_name=add_label_name)
