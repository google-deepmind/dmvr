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

"""Utils for processing datasets features."""

from typing import Any, Optional, Sequence, Union

from dmvr import builders
from dmvr import tokenizers
import tensorflow as tf


# ----------------------------------------------------------------------
# ----------------------------- Utilities. -----------------------------
# ----------------------------------------------------------------------


def _get_random_sampling_offset(sequence: tf.Tensor,
                                num_steps: int,
                                stride: int,
                                seed: Optional[int] = None) -> tf.Tensor:
  """Calculates the initial offset for a sequence where all steps will fit.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    stride: Distance to sample between timesteps.
    seed: A deterministic seed to use when sampling.

  Returns:
    The first index to begin sampling from. A best effort is made to provide a
    starting index such that all requested steps fit within the sequence (i.e.
    `offset + 1 + (num_steps - 1) * stride` < len(sequence)`). If this is not
    satisfied, the starting index is always 0.
  """
  sequence_length = tf.shape(input=sequence)[0]
  max_offset = tf.maximum(sequence_length - (num_steps - 1) * stride, 1)
  return tf.random.uniform((),
                           maxval=tf.cast(max_offset, dtype=tf.int32),
                           dtype=tf.int32,
                           seed=seed)


def sample_or_pad_sequence_indices(sequence: tf.Tensor, num_steps: int,
                                   repeat_sequence: bool, stride: int,
                                   offset: int) -> tf.Tensor:
  """Returns indices to take for sampling or padding a sequence to fixed size.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    repeat_sequence: A boolean indicates whether the sequence will repeat to
      have enough steps for sampling. If `False`, a runtime error is thrown if
      `num_steps` * `stride` is longer than sequence length.
    stride: Distance to sample between timesteps.
    offset: Offset(s) to be used for sampling.

  Returns:
    Indices to gather from the sequence tensor to get a fixed size sequence.
  """
  sequence_length = tf.shape(input=sequence)[0]
  sel_idx = tf.range(sequence_length)

  if repeat_sequence:
    # Repeats sequence until `num_steps` are available in total.
    num_repeats = tf.cast(
        tf.math.ceil(
            tf.divide(
                tf.cast(num_steps * stride + offset, dtype=tf.float32),
                tf.cast(sequence_length, dtype=tf.float32))), dtype=tf.int32)
    sel_idx = tf.tile(sel_idx, [num_repeats])
  steps = tf.range(offset, offset + num_steps * stride, stride)

  return tf.gather(sel_idx, steps)


# ----------------------------------------------------------------------
# ----------------- Methods used in sample functions. ------------------
# ----------------------------------------------------------------------


def sample_linspace_sequence(sequence: tf.Tensor,
                             num_windows: int,
                             num_steps: int,
                             stride: int = 1) -> tf.Tensor:
  """Samples `num_windows` segments from sequence with linearly spaced offsets.

  The samples are concatenated in a single Tensor in order to have the same
  format structure per timestep (e.g. a single frame). If `num_steps` * `stride`
  is bigger than the number of timesteps, the sequence is repeated. This
  function can be used in evaluation to extract enough segments in order to span
  the entire sequence.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_windows: Number of windows to be retrieved from the sequence.
    num_steps: Number of steps (e.g. frames) to take in each window.
    stride: Distance to sample between timesteps.

  Returns:
    A single tensor with first dimension `num_windows` * `num_steps`. The tensor
    contains the concatenated list of `num_windows` tensors which offsets have
    been linearly spaced from input.
  """
  sequence_length = tf.shape(input=sequence)[0]
  max_offset = tf.maximum(0, sequence_length - num_steps * stride)
  offsets = tf.linspace(0.0, tf.cast(max_offset, tf.float32), num_windows)
  offsets = tf.cast(offsets, tf.int32)

  all_indices = []
  for i in range(num_windows):
    all_indices.append(
        sample_or_pad_sequence_indices(
            sequence=sequence,
            num_steps=num_steps,
            repeat_sequence=True,  # Will repeat the sequence if request more.
            stride=stride,
            offset=offsets[i]))

  indices = tf.concat(all_indices, axis=0)
  indices.set_shape((num_windows * num_steps,))
  output = tf.gather(sequence, indices)

  return output


def sample_sequence(
    sequence: tf.Tensor,
    num_steps: int,
    random: bool,
    stride: int = 1,
    seed: Optional[int] = None,
    state: Optional[builders.ProcessorState] = None) -> tf.Tensor:
  """Samples a single segment of size `num_steps` from a given sequence.

  If `random` is not `True`, this function will simply sample the central window
  of the sequence. Otherwise, a random offset will be chosen in a way that the
  desired `num_steps` might be extracted from the sequence.

  In order to keep coherence among different sequences sampled using random true
  (e.g. image and audio), an optional state is accepted as parameter and used to
  keep track of the first offset, using a proportional offset to sample from the
  second sequence.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    num_steps: Number of steps (e.g. frames) to take.
    random: A boolean indicating whether to random sample the single window. If
      `True`, the offset is randomized. If `False`, the middle frame minus half
      of `num_steps` is the first frame.
    stride: Distance to sample between timesteps.
    seed: A deterministic seed to use when sampling.
    state: A mutable dictionary where keys are strings. The dictionary might
      contain 'sample_offset_proportion' as key with metadata useful for
      sampling. It will be modified with added metadata if needed. This can be
      used to keep consistency between sampling of different sequences.

  Returns:
    A single tensor with first dimension `num_steps` with the sampled segment.
  """
  sequence_length = tf.shape(input=sequence)[0]
  sequence_length = tf.cast(sequence_length, tf.float32)

  if random:
    if state and 'sample_offset_proportion' in state:
      # Read offset from state to ensure consistent offsets for different
      # modalities.
      offset = state['sample_offset_proportion'] * sequence_length
      offset = tf.cast(tf.math.round(offset), tf.int32)
    else:
      offset = _get_random_sampling_offset(
          sequence=sequence,
          num_steps=num_steps,
          stride=stride,
          seed=seed)

      if state is not None:
        # Update state.
        sample_offset_proportion = tf.cast(offset, tf.float32) / sequence_length
        state['sample_offset_proportion'] = sample_offset_proportion

  else:
    offset = tf.maximum(
        0, tf.cast((sequence_length - num_steps * stride) // 2, tf.int32))

  indices = sample_or_pad_sequence_indices(
      sequence=sequence,
      num_steps=num_steps,
      repeat_sequence=True,  # Will repeat the sequence if request more.
      stride=stride,
      offset=offset)
  indices.set_shape((num_steps,))
  output = tf.gather(sequence, indices)

  return output


def sample_or_pad_non_sorted_sequence(
    sequence: tf.Tensor,
    max_num_steps: int,
    pad_value: Any,
    random: bool,
    seed: Optional[int] = None,
    state: Optional[builders.ProcessorState] = None,
    state_key: str = 'sample_sequence_random_perm') -> tf.Tensor:
  """Samples or pads (with `pad_value`) elements from the input sequence.

  The input sequence can be multidimensional, but the sampling or pads will
  only happen in the first dimension.

  Args:
    sequence: Any tensor where the first dimension is timesteps.
    max_num_steps: Maximum number of steps to be kept from the input. If the
      input contains more, it's sampled, if less, it's padded.
    pad_value: Value to be used when padding. Same type as `sequence`.
    random: A boolean indicating whether to random sample from the input. If
      `True`, a random permutation is taken. If `False`, the first
      `max(max_num_steps, sequence_length)` elements are taken.
    seed: A deterministic seed to use when sampling.
    state:  A mutable dictionary where keys are strings. The dictionary might
      contain an entry with `state_key` as key with metadata useful for
      sampling. It will be modified with added metadata if needed. This can be
      used to keep consistency between sampling of different sequences. Note
      that a runtime error will be raised in case state is provided but the
      sequences that one tries to sync are of different lenghts.
    state_key: Name of the state entry that controls the random sampling.

  Returns:
    A single tensor with first dimension `max_num_steps` with the sampled
    elements.

  Raises:
    tf.errors.InvalidArgumentError: if state is provided but the sequences that
      one tries to sync are of different lengths.
  """
  sequence_length = tf.shape(input=sequence)[0]
  if random:
    if state and state_key in state:
      # Read offset from state to ensure consistent offsets for different
      # modalities.
      random_perm = state[state_key]
      tf.debugging.assert_equal(
          sequence_length, tf.shape(input=random_perm)[0],
          ('Trying to sync the sampling of two sequences that do not have the '
           'same number of elements!'))
    else:
      random_perm = tf.argsort(tf.random.uniform((sequence_length,), seed=seed))
      if state is not None:
        state[state_key] = random_perm
    sequence = tf.gather(sequence, random_perm)

  padding_pattern = [[0, tf.maximum(0, max_num_steps - sequence_length)],]
  num_dim = len(tf.shape(input=sequence))
  if num_dim > 1:
    padding_pattern.extend([[0, 0]] * (num_dim - 1))
  return tf.pad(
      tensor=sequence[:max_num_steps],
      paddings=padding_pattern,
      constant_values=pad_value)


# ----------------------------------------------------------------------
# ----------------- Methods used in decode functions. ------------------
# ----------------------------------------------------------------------


def decode_jpeg(image_string: tf.Tensor, channels: int = 0) -> tf.Tensor:
  """Decodes JPEG raw bytes string into a RGB uint8 tensor.

  Args:
    image_string: A tensor of type strings with the raw JPEG bytes where the
      first dimension is timesteps.
    channels: Number of channels of the JPEG image. Allowed values are 0, 1 and
      3. If 0, the number of channels will be calculated at runtime and no
      static shape is set.

  Returns:
    A `tf.Tensor` of shape [T, H, W, C] of type `tf.uint8` with the decoded
    images.
  """
  return tf.map_fn(
      lambda x: tf.image.decode_jpeg(x, channels=channels),
      image_string, back_prop=False, dtype=tf.uint8)


# ----------------------------------------------------------------------
# --------------- Methods used in preprocess functions. ----------------
# ----------------------------------------------------------------------


def set_shape(
    inputs: tf.Tensor,
    shape: Union[tf.TensorShape, Sequence[Optional[int]]]) -> tf.Tensor:
  """Sets the shape of the given tensor and returns it."""
  inputs.set_shape(shape)
  return inputs


def crop_image(frames: tf.Tensor,
               height: int,
               width: int,
               random: bool = False,
               seed: Optional[int] = None,
               state: Optional[builders.ProcessorState] = None) -> tf.Tensor:
  """Crops the images in the given sequence of images.

  If requested size is bigger than image size, image is padded with 0. If not
  random cropping, a central crop is performed.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    height: Cropped image height.
    width: Cropped image width.
    random: A boolean indicating if crop should be randomized.
    seed: A deterministic seed to use when random cropping.
    state: A mutable dictionary where keys are strings. The dictionary might
      contain 'crop_offset_proportion' as key with metadata useful for cropping.
      It will be modified with added metadata if needed. This can be used to
      keep consistency between cropping of different sequences of images.

  Returns:
    A tensor of shape [timesteps, output_h, output_w, channels] of same type as
    input with the cropped images.
  """
  if random:
    # Random spatial crop. tf.image.random_crop is not used since the offset is
    # needed to ensure consistency between crops on different modalities.
    shape = tf.shape(input=frames)
    # If a static_shape is available (e.g. when using this method from add_image
    # method), it will be used to have an output tensor with static shape.
    static_shape = frames.shape.as_list()
    seq_len = shape[0] if static_shape[0] is None else static_shape[0]
    channels = shape[3] if static_shape[3] is None else static_shape[3]
    size = tf.convert_to_tensor(value=(seq_len, height, width, channels))

    if state and 'crop_offset_proportion' in state:
      # Use offset set by a previous cropping: [0, offset_h, offset_w, 0].
      offset = state['crop_offset_proportion'] * tf.cast(shape, tf.float32)
      offset = tf.cast(tf.math.round(offset), tf.int32)
    else:
      # Limit of possible offset in order to fit the entire crop:
      # [1, input_h - target_h + 1, input_w - target_w + 1, 1].
      limit = shape - size + 1
      offset = tf.random.uniform(
          shape=(4,),
          dtype=tf.int32,
          maxval=tf.int32.max,
          seed=seed) % limit  # [0, offset_h, offset_w, 0]

      if state is not None:
        # Update state.
        offset_proportion = tf.cast(offset, tf.float32) / tf.cast(
            shape, tf.float32)
        state['crop_offset_proportion'] = offset_proportion

    frames = tf.slice(frames, offset, size)
  else:
    # Central crop or pad.
    frames = tf.image.resize_with_crop_or_pad(frames, height, width)
  return frames


def resize_smallest(frames: tf.Tensor,
                    min_resize: int,
                    is_flow: bool = False,
                    method: str = tf.image.ResizeMethod.BILINEAR) -> tf.Tensor:
  """Resizes frames so that `min(height, width)` is equal to `min_resize`.

  This function will do nothing if the `min(height, width)` is already equal to
  `min_resize`. This allows to save compute time.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    min_resize: Minimum size of the final image dimensions.
    is_flow: If is flow, will modify the raw values to account for the resize.
      For example, if the flow image is resized by a factor k, we need to
      multiply the flow values by the same factor k since one pixel displacement
      in the resized image corresponds to only 1/k pixel displacement in the
      original image.
    method: A resizing method.

  Returns:
    A tensor of shape [timesteps, output_h, output_w, channels] of same type as
    input, where `min(output_h, output_w)` is `min_resize`.
  """
  if is_flow and frames.dtype != tf.float32:
    raise ValueError('If `is_flow`, frames should be given in `tf.float32`.')
  shape = tf.shape(input=frames)
  input_h = shape[1]
  input_w = shape[2]

  output_h = tf.maximum(min_resize, (input_h * min_resize) // input_w)
  output_w = tf.maximum(min_resize, (input_w * min_resize) // input_h)

  def resize_fn():
    frames_resized = tf.image.resize(
        frames, (output_h, output_w), method=method)
    return tf.cast(frames_resized, frames.dtype)

  should_resize = tf.math.logical_or(tf.not_equal(input_w, output_w),
                                     tf.not_equal(input_h, output_h))
  frames = tf.cond(
      pred=should_resize, true_fn=resize_fn, false_fn=lambda: frames)

  if is_flow:
    # Apply a multiplier to keep the right magnitude in the flow.
    frames = frames * tf.cast(output_h / input_h, tf.float32)

  return frames


def random_flip_left_right(frames: tf.Tensor,
                           seed: Optional[int] = None,
                           state: Optional[builders.ProcessorState] = None,
                           is_flow: bool = False) -> tf.Tensor:
  """Flips all the frames (consistently) with a probability of 50%.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    seed: A seed to use for the random sampling.
    state: A mutable dictionary where keys are strings. The dictionary might
      contain 'flip_left_right_is_flipped' as key with metadata useful for
      flipping. It will be modified with added metadata if needed. This can be
      used to keep consistency between flipping of different sequences of
      images.
    is_flow: If is flow and the image is flipped, the horizontal component
      of the flow will be multiplied by -1 to account for the symmetry.

  Returns:
    A tensor of shape [timesteps, output_h, output_w, channels] eventually
    flipped left right.
  """
  if state and 'flip_left_right_is_flipped' in state:
    is_flipped = state['flip_left_right_is_flipped']
  else:
    is_flipped = tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32,
                                   seed=seed)
    if state is not None:
      # Update state.
      state['flip_left_right_is_flipped'] = is_flipped

  frames = tf.cond(pred=tf.equal(is_flipped, 1),
                   true_fn=lambda: tf.image.flip_left_right(frames),
                   false_fn=lambda: frames)

  if is_flow:
    # Multiply horizontal component by -1.0 if `is_flipped`.
    channel_mult = tf.constant([-1.0, 1.0, 1.0])[None, None, None, :]
    frames = tf.cond(pred=tf.equal(is_flipped, 1),
                     true_fn=lambda: channel_mult * frames,
                     false_fn=lambda: frames)

  return frames


def normalize_image(frames: tf.Tensor,
                    zero_centering_image: bool,
                    dtype: tf.dtypes.DType = tf.float32) -> tf.Tensor:
  """Normalizes images.

  Args:
    frames: A tensor of numbers.
    zero_centering_image: If `True`, results are in [-1, 1], if `False`, results
      are in [0, 1].
    dtype: Type of output tensor.

  Returns:
    A Tensor of same shape as the input and of the given type.
  """
  frames = tf.cast(frames, dtype)
  if zero_centering_image:
    frames = frames * (2.0 / 255.0) - 1.0
  else:
    frames /= 255.0
  return frames


def scale_jitter_augm(
    frames: tf.Tensor,
    min_scale_factor: float = 0.8,
    max_scale_factor: float = 1.2,
    prob: float = 0.8,
    seed: Optional[int] = None,
    state: Optional[builders.ProcessorState] = None,
    is_flow: bool = False,
    method: str = tf.image.ResizeMethod.BILINEAR,
) -> tf.Tensor:
  """Applies scale jitter to videos with probability `prob`.

  In details this will independently sample a factor along the height and the
  width of the frames and rescale the video accordingly.

  Args:
    frames: A tensor of dimension [timesteps, input_h, input_w, channels].
    min_scale_factor: Minimum scale factor to sample.
    max_scale_factor: Maximum scale factor to sample.
    prob: The probability that the scale of the video is going to be jittered.
    seed: A seed to use for the random sampling.
    state: A mutable dictionary where keys are strings. The dictionary might
      contain 'scale_jitter_augm_info' as key with metadata useful for
      jittering. It will be modified with added metadata if needed. This can be
      used to keep consistency between jittering of different sequences of
      images.
    is_flow: If is flow, will modify the raw values to account for the resize.
      For example, if the flow image is resized by a factor k, we need to
      multiply the flow values by the same factor k since one pixel displacement
      in the resized image corresponds to only 1/k pixel displacement in the
      original image.
    method: A resizing method.

  Returns:
    A tensor of shape [timesteps, output_h, output_w, channels] which spatial
    dimensions have eventually been modified with the same type as the input.
  """
  if not 0. <= prob <= 1.0:
    raise ValueError(f'`prob` should be in [0, 1] but {prob} was given.')

  def scale_augment(frames: tf.Tensor,
                    h_scale: tf.float32,
                    w_scale: tf.float32) -> tf.Tensor:
    """Do scale jitter."""
    _, input_height, input_width, _ = tf.unstack(tf.shape(input=frames))
    rdm_resize_height = tf.cast(
        h_scale * tf.cast(input_height, tf.float32), tf.int32)
    rdm_resize_width = tf.cast(
        w_scale * tf.cast(input_width, tf.float32), tf.int32)
    resize_shape = tf.stack([rdm_resize_height, rdm_resize_width])
    frames = tf.cast(
        tf.image.resize(frames, resize_shape, method=method),
        frames.dtype)
    if is_flow:
      channel_mult = tf.stack([h_scale, w_scale, 1.0])[None, None, None, :]
      # Apply a multiplier to keep the right magnitude in the flow.
      frames = frames * channel_mult
    return frames

  if state and 'scale_jitter_augm_info' in state:
    h_scale, w_scale, coin_toss = state['scale_jitter_augm_info']
  else:
    h_scale = tf.random.uniform(
        [], minval=min_scale_factor, maxval=max_scale_factor, dtype=tf.float32,
        seed=seed)
    w_scale = tf.random.uniform(
        [], minval=min_scale_factor, maxval=max_scale_factor, dtype=tf.float32,
        seed=seed)
    coin_toss = tf.random.uniform(
        [], minval=0, maxval=1, dtype=tf.float32, seed=seed)

    if state is not None:
      # Update state.
      state['scale_jitter_augm_info'] = (h_scale, w_scale, coin_toss)

  frames = tf.cond(
      pred=tf.less(coin_toss, tf.cast(prob, tf.float32)),
      true_fn=lambda: scale_augment(frames, h_scale=h_scale, w_scale=w_scale),
      false_fn=lambda: frames)

  return frames


def color_default_augm(frames: tf.Tensor,
                       zero_centering_image: bool = False,
                       prob_color_augment: float = 0.8,
                       prob_color_drop: float = 0.0,
                       seed: Optional[int] = None):
  """Standard color augmentation for videos.

  Args:
    frames: A float32 tensor of shape [timesteps, input_h, input_w, channels].
    zero_centering_image: If `True`, results are in [-1, 1], if `False`, results
      are in [0, 1].
    prob_color_augment: Probability of applying color augmentation.
    prob_color_drop: Probability of droping the colors to gray scale.
    seed: A seed to use for the random sampling.

  Returns:
    A tensor of same shape as the input with color eventually altered.
  """
  if frames.dtype != tf.float32:
    raise ValueError(f'`frames` should be in float32 (but was {frames.dtype}).')

  if not 0. <= prob_color_augment <= 1.0:
    raise ValueError(
        f'`prob_color_augment` ({prob_color_augment} given) should be in '
        '[0, 1].')

  if not 0. <= prob_color_drop <= 1.0:
    raise ValueError(
        f'`prob_color_drop` ({prob_color_drop} given) should be in [0, 1].')

  def color_augment(video: tf.Tensor) -> tf.Tensor:
    """Do standard color augmentations."""
    # Note the same augmentation will be applied to all frames of the video.
    if zero_centering_image:
      video = 0.5 * (video + 1.0)
    video = tf.image.random_brightness(video, max_delta=32. / 255.)
    video = tf.image.random_saturation(video, lower=0.6, upper=1.4)
    video = tf.image.random_contrast(video, lower=0.6, upper=1.4)
    video = tf.image.random_hue(video, max_delta=0.2)
    video = tf.clip_by_value(video, 0.0, 1.0)
    if zero_centering_image:
      video = 2 * (video-0.5)
    return video

  def color_drop(video: tf.Tensor) -> tf.Tensor:
    """Do color drop."""
    video = tf.image.rgb_to_grayscale(video)
    video = tf.tile(video, [1, 1, 1, 3])
    return video

  # Eventually applies color augmentation.
  coin_toss_color_augment = tf.random.uniform(
      [], minval=0, maxval=1, dtype=tf.float32, seed=seed)
  frames = tf.cond(
      pred=tf.less(coin_toss_color_augment,
                   tf.cast(prob_color_augment, tf.float32)),
      true_fn=lambda: color_augment(frames),
      false_fn=lambda: frames)

  # Eventually applies color drop.
  coin_toss_color_drop = tf.random.uniform(
      [], minval=0, maxval=1, dtype=tf.float32, seed=seed)
  frames = tf.cond(
      pred=tf.less(coin_toss_color_drop, tf.cast(prob_color_drop, tf.float32)),
      true_fn=lambda: color_drop(frames),
      false_fn=lambda: frames)

  return frames


def space_to_depth(frames: tf.Tensor,
                   temporal_block_size: int = 1,
                   spatial_block_size: int = 1) -> tf.Tensor:
  """Performs per frame space to depth.

  Args:
    frames: A tensor of dimension [T, H, W, C].
    temporal_block_size: Size of the block for temporal dimension.
    spatial_block_size: Size of the block for spatial dimensions.

  Returns:
    A tensor of shape [T / t_b, H / s_b, W / s_b, t_b * s_b * s_b * C] with the
    same type as the input, where t_b is the `temporal_block_size` and s_b is
    the `spatial_block_size`.
  """
  t, h, w, c = frames.shape.as_list()
  frames = tf.reshape(frames, (
      t // temporal_block_size, temporal_block_size, h // spatial_block_size,
      spatial_block_size, w // spatial_block_size, spatial_block_size, c))
  frames = tf.transpose(a=frames, perm=(0, 2, 4, 1, 3, 5, 6))
  frames = tf.reshape(frames, (
      t // temporal_block_size, h // spatial_block_size,
      w // spatial_block_size,
      temporal_block_size * (spatial_block_size ** 2) * c))
  return frames


def crop_or_pad_words(words: tf.Tensor,
                      max_num_words: int,
                      pad_value: int = 0) -> tf.Tensor:
  """Crop or pad given sequence of word indices.

  Args:
    words: Tensor of shape [T, sentence_length] of word indices.
    max_num_words: Maximum number of words in final result.
    pad_value: Value to be used in paddings.

  Returns:
    A Tensor of shape [T, max_num_words].
  """
  num_words = tf.shape(input=words)[1]
  words = tf.pad(
      tensor=words[:, :max_num_words],
      paddings=((0, 0), (0, tf.maximum(0, max_num_words - num_words))),
      constant_values=pad_value)
  words.set_shape((None, max_num_words))
  return words


def tokenize(features: builders.FeaturesDict,
             tokenizer: tokenizers.TextTokenizer,
             raw_string_name: str,
             tokenized_name: str,
             prepend_bos: bool,
             append_eos: bool,
             max_num_tokens: int,
             keep_raw_string: bool) -> builders.FeaturesDict:
  """Tokenize raw string with tokenizer.

  Args:
    features: A dictionary of features.
    tokenizer: An instance of a text tokenizer.
    raw_string_name: The name of the raw string feature in features.
    tokenized_name: The name of the desired tokenized feature in the output.
    prepend_bos: Whether to prepend BOS in the tokenizer.
    append_eos: Whether to append EOS in the tokenizer.
    max_num_tokens: Number of tokens in final result. The tokenized sentence
      will be either crop or padded using the tokenizer pad token ID.
    keep_raw_string: Whether to keep the raw string in the output.

  Returns:
    A FeaturesDict containing the tokenized string.
  """
  raw_caption = features[raw_string_name]
  tokenized = tokenizer.string_tensor_to_indices(
      raw_caption, prepend_bos=prepend_bos, append_eos=append_eos,
      max_num_tokens=max_num_tokens)
  if not keep_raw_string:
    del features[raw_string_name]
  features[tokenized_name] = tokenized
  return features


def _preemphasis(audio: tf.Tensor, coef: float = 0.97) -> tf.Tensor:
  """Scale up the high frequency components in the waveform.

  Args:
    audio: Input waveform.
    coef: Pre-emphasis coefficient.

  Returns:
    Pre-emphasized audio.
  """
  return tf.concat([audio[:1], audio[1:] - coef * audio[:-1]], axis=0)


def compute_audio_spectrogram(
    features: builders.FeaturesDict,
    num_subclips: int = 1,
    sample_rate: int = 48000,
    spectrogram_type: str = 'logmf',
    frame_length: int = 2048,
    frame_step: int = 1024,
    num_features: int = 80,
    lower_edge_hertz: float = 80.0,
    upper_edge_hertz: float = 7600.0,
    preemphasis: Optional[float] = None,
    normalize: bool = False,
    audio_feature_name: str = builders.AUDIO_MEL_FEATURE_NAME,
    spectrogram_feature_name: str = builders.AUDIO_MEL_FEATURE_NAME,
    fft_output_conversion: str = 'magnitude',
    ) -> builders.FeaturesDict:
  """Computes audio spectrograms.

  Args:
    features: A dictionary of features.
    num_subclips: Number of test clips (1 by default). If more than 1, this will
      sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggreagated in the batch dimension.
    sample_rate: The sample rate of the input audio.
    spectrogram_type: The type of the spectrogram to be extracted from the
      waveform. Can be either `spectrogram`, `logmf`, and `mfcc`.
    frame_length: The length of each spectroram frame.
    frame_step: The stride of spectrogram frames.
    num_features: The number of spectrogram features.
    lower_edge_hertz: Lowest frequency to consider.
    upper_edge_hertz: Highest frequency to consider.
    preemphasis: The strength of pre-emphasis on the waveform. If None, no
      pre-emphasis will be applied.
    normalize: Whether to normalize the waveform or not.
    audio_feature_name: The name of the raw audio feature in features.
    spectrogram_feature_name: The name of the spectrogram feature in features.
    fft_output_conversion: The string indicating the output conversion function.
      Currently, only `magnitude` and `magnitude_squared` are supported.

  Returns:
    A FeaturesDict containing the extracted spectrograms.

  Raises:
    ValueError: if `spectrogram_type` is one of `spectrogram`, `logmf`, or
      `mfcc`.
  """
  if spectrogram_type not in ['spectrogram', 'logmf', 'mfcc']:
    raise ValueError('Spectrogram type should be one of `spectrogram`, '
                     '`logmf`, or `mfcc`, got {}'.format(spectrogram_type))

  if fft_output_conversion not in ['magnitude', 'magnitude_squared']:
    raise ValueError(
        'FFT output conversion should be one of `magnitude` or '
        '`magnitude_squared, god {}`'.format(fft_output_conversion))

  raw_audio = features[audio_feature_name]
  if normalize:
    raw_audio /= (
        tf.reduce_max(tf.abs(raw_audio), axis=-1, keepdims=True) + 1e-8)
    features[audio_feature_name] = raw_audio
  if num_subclips > 1:
    raw_audio = tf.reshape(raw_audio, [num_subclips, -1])
  if preemphasis is not None:
    raw_audio = _preemphasis(raw_audio, preemphasis)

  def _extract_spectrogram(
      waveform: tf.Tensor,
      spectrogram_type: str) -> tf.Tensor:
    stfts = tf.signal.stft(waveform,
                           frame_length=frame_length,
                           frame_step=frame_step,
                           fft_length=frame_length,
                           window_fn=tf.signal.hann_window,
                           pad_end=True)
    if fft_output_conversion == 'magnitude_squared':
      stfts = tf.square(stfts)
    spectrograms = tf.abs(stfts)

    if spectrogram_type == 'spectrogram':
      return spectrograms[..., :num_features]

    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_features, num_spectrogram_bins, sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    if spectrogram_type == 'logmf':
      return log_mel_spectrograms

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :13]
    return mfccs

  spectrogram = _extract_spectrogram(raw_audio, spectrogram_type)
  features[spectrogram_feature_name] = spectrogram
  return features


def _resample_audio_fft(
    x: tf.Tensor,
    in_sample_rate: int,
    out_sample_rate: int,
    resolution_bits: Optional[float] = None) -> tf.Tensor:
  """Resample audio using FFTs.

  Args:
    x: Input audio signal.
    in_sample_rate: The original sample rate of the input audio.
    out_sample_rate: The target sample rate.
    resolution_bits: Resolution bits used to scale the FFTs. If None no scaling
      is used.

  Returns:
    The resampled audio signal.
  """
  axis = -1  # tf.signal.fft operates on the innermost dimension of x
  if in_sample_rate == out_sample_rate:
    return x

  scale = 2**(resolution_bits - 1) if resolution_bits else None

  if scale:
    x /= scale

  factor = float(out_sample_rate) / in_sample_rate
  original_size = tf.shape(x)[axis]
  resampled_size = tf.cast(
      tf.cast(original_size, dtype=tf.float32) * factor, dtype=tf.int32)

  x_ = tf.signal.fft(tf.cast(x, dtype=tf.complex64))

  shape = x.get_shape().as_list()
  rank = len(shape)
  sl_beg = [slice(None)] * rank
  sl_end = [slice(None)] * rank

  min_size = tf.minimum(resampled_size, original_size)
  sl_beg[axis] = slice(0, (min_size + 1) // 2)
  sl_end[axis] = slice(-(min_size - 1) // 2, None)

  # Compute padding: empty unless upsampling (resampled_size > original_size).
  pad_shape = list(shape)
  pad_shape[axis] = tf.maximum(0, resampled_size - original_size)
  padding = tf.zeros(pad_shape, dtype=x_.dtype)

  y_ = tf.concat([x_[sl_beg], padding, x_[sl_end]], axis=axis)
  y = tf.signal.ifft(y_)
  y = tf.math.real(y) * factor

  # Deliberately subtract 1 to prevent clipped values from going out of range.
  y = tf.clip_by_value(y, -1, 1)
  if scale:
    y *= scale - 1
  if shape[axis] is not None:
    shape[axis] = int(shape[axis] * factor)
  y.set_shape(shape)

  return y


def resample_audio(
    audio: tf.Tensor,
    in_sample_rate: int,
    out_sample_rate: int,
    is_training: bool = True,
    num_subclips: int = 1,
    ) -> tf.Tensor:
  """Resamples raw audio.

  Args:
    audio: Input audio signal.
    in_sample_rate: The original sample rate of the input audio.
    out_sample_rate: The target sample rate.
    is_training: If the current stage is training.
    num_subclips: Number of test clips (1 by default). If more than 1, this will
      sample multiple linearly spaced clips within each audio at test time.
      If 1, then a single clip in the middle of the audio is sampled. The clips
      are aggreagated in the batch dimension.

  Returns:
    The resampled audio signal.
  """
  if num_subclips > 1 and not is_training:
    audio = tf.reshape(audio, [num_subclips, -1])
  return _resample_audio_fft(audio, in_sample_rate, out_sample_rate)


# ----------------------------------------------------------------------
# --------------- Methods used in postprocess functions. ---------------
# ----------------------------------------------------------------------


def batched_video_transpose(batched_img: tf.Tensor,
                            perm: Sequence[int]) -> tf.Tensor:
  """Transposes the given Tensor (used to transpose on host instead of TPU)."""
  return tf.transpose(a=batched_img, perm=perm)


def batched_space_to_depth(frames: tf.Tensor,
                           temporal_block_size: int = 1,
                           spatial_block_size: int = 1) -> tf.Tensor:
  """Performs per batch space to depth.

  Args:
    frames: A tensor of dimension [B, T, H, W, C].
    temporal_block_size: Size of the block for temporal dimension.
    spatial_block_size: Size of the block for spatial dimensions.

  Returns:
    A tensor of shape [B, T / t_b, H / s_b, W / s_b, t_b * s_b * s_b * C] with
    the same type as the input, where t_b is the `temporal_block_size` and s_b
    is the `spatial_block_size`.
  """
  _, t, h, w, c = frames.shape.as_list()
  frames = tf.reshape(frames, (
      -1, t // temporal_block_size, temporal_block_size,
      h // spatial_block_size, spatial_block_size, w // spatial_block_size,
      spatial_block_size, c))
  frames = tf.transpose(a=frames, perm=(0, 1, 3, 5, 2, 4, 6, 7))
  frames = tf.reshape(frames, (
      -1, t // temporal_block_size, h // spatial_block_size,
      w // spatial_block_size,
      temporal_block_size * (spatial_block_size ** 2) * c))
  return frames
