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
"""A simple tokenizer interface with basic implementations."""

import abc
from typing import Optional, Sequence, Union

import clip.simple_tokenizer
import tensorflow as tf
import tensorflow_text

import sentencepiece as spm


class TextTokenizer(abc.ABC):
  """Base class for text tokenizers."""

  def initialize(self):
    """Initializes tensorflow tables and models."""
    return

  @abc.abstractmethod
  def string_tensor_to_indices(self,
                               string_tensor: Union[tf.Tensor, Sequence[str]],
                               prepend_bos: bool = False,
                               append_eos: bool = False,
                               max_num_tokens: Optional[int] = 32) -> tf.Tensor:
    """Tokenizes input text, mapping a tensor of strings to a tensor of ints.

    Args:
      string_tensor: Input string tensor of shape [num_texts].
      prepend_bos: Whether to prepend the BOS (beginning of sentence) token to
        the output tokens.
      append_eos: Whether to append the EOS (end of sentence) token to the
        output tokens.
      max_num_tokens: Maximum number of tokens to return per caption. If
        provided, the tokens will be padded / cut at the given size. If not, a
        tensor of unknown size will be returned.

    Returns:
      A `tf.int32` tensor of shape [num_texts, `max_num_tokens`] if
        `max_num_tokens` is provided or [num_texts, max_num_tokens_in_batch]
        otherwise.
    """

  @abc.abstractmethod
  def indices_to_string(self, indices: Sequence[int]) -> str:
    """Detokenizes, mapping a python sequence of indices to a string."""

  @property
  @abc.abstractmethod
  def vocab_size(self) -> int:
    """Returns the vocabulary size."""

  @property
  @abc.abstractmethod
  def pad_token(self) -> int:
    """Returns index of the PAD token."""

  @property
  @abc.abstractmethod
  def bos_token(self) -> int:
    """Returns index of the BOS token."""

  @property
  @abc.abstractmethod
  def eos_token(self) -> int:
    """Returns index of the EOS token."""

  @property
  @abc.abstractmethod
  def unk_token(self) -> int:
    """Returns index of the UNK token."""


class SentencePieceTokenizer(TextTokenizer):
  """SentencePiece tokenizer from a pre-trained SentencePiece model.

  Pre-trained models are provided in multiple repositories around the web. See
  https://github.com/google/sentencepiece for info on how to train new models on
  specific corpus.
  """

  def __init__(self, model_path: str):
    """Initializes the `SentencePieceTokenizer`.

    Args:
      model_path: Path to the '.model' file.
    """
    self._model_path = model_path
    self._sp_model = spm.SentencePieceProcessor()
    self._sp_model.Load(model_path)

    self._vocab_size = self._sp_model.GetPieceSize()
    self._bos_token = self._sp_model.bos_id()
    self._eos_token = self._sp_model.eos_id()
    self._pad_token = self._sp_model.pad_id()
    self._unk_token = self._sp_model.unk_id()

    self._tf_sp_model = None

  def initialize(self):
    with tf.io.gfile.GFile(self._model_path, 'rb') as f:
      self._tf_sp_model = tensorflow_text.SentencepieceTokenizer(
          model=f.read(), out_type=tf.int32, add_bos=True, add_eos=True)

  def string_tensor_to_indices(self,
                               string_tensor: Union[tf.Tensor, Sequence[str]],
                               prepend_bos: bool = False,
                               append_eos: bool = False,
                               max_num_tokens: Optional[int] = 32) -> tf.Tensor:
    if self._tf_sp_model is None:
      raise RuntimeError('Model was not initialized. Call `initialize` method.')

    tokenized = self._tf_sp_model.tokenize(string_tensor)
    tokenized = tokenized if prepend_bos else tokenized[..., 1:]
    tokenized = tokenized if append_eos else tokenized[..., :-1]

    # Pad to `max_num_tokens`.
    shape = None if max_num_tokens is None else [None, max_num_tokens]
    tokenized = tokenized.to_tensor(default_value=self._pad_token, shape=shape)
    return tokenized

  def indices_to_string(self, indices: Sequence[int]) -> str:
    return self._sp_model.DecodeIds(indices)

  def string_to_indices(self,
                        string: str,
                        prepend_bos: bool = False,
                        append_eos: bool = False,
                        max_num_tokens: Optional[int] = 32) -> Sequence[int]:
    """Tokenizes, mapping a python string to a sequence of indices."""
    tokenized = self._sp_model.EncodeAsIds(string)
    tokenized = [self._bos_token] * prepend_bos + tokenized
    tokenized += [self._eos_token] * append_eos
    if max_num_tokens:
      tokenized = tokenized[:max_num_tokens]
      num_tokens = len(tokenized)
      tokenized = tokenized + [self._pad_token] * (max_num_tokens - num_tokens)
    return tokenized

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def pad_token(self):
    return self._pad_token

  @property
  def bos_token(self):
    return self._bos_token

  @property
  def eos_token(self):
    return self._eos_token

  @property
  def unk_token(self):
    return self._unk_token


class WordTokenizer(TextTokenizer):
  """Vocabulary based word tokenizer."""

  PAD = '<pad>'
  BOS = '<bos>'
  EOS = '<eos>'
  UNK = '<unk>'

  def __init__(self, vocabulary_path: str):
    """Initializes the `WordTokenizer`.

    Args:
      vocabulary_path: A path to a vocabulary file. The vocabulary is a simple
        text file where each line is of the form: 'idx_word word' or simply
        'word' (the line index will be used). The vocabulary should at least
        contain the words: '<pad>', '<bos>', '<eos>' and '<unk>'.
    """
    # Parse the vocabulary. The expected format is either one word per line (and
    # the index for that word will be the line index) or an index and a word,
    # split by space.
    idx2word = {}
    with tf.io.gfile.GFile(vocabulary_path) as f:
      for line_idx, line in enumerate(f):
        line = line.strip().split(' ')

        if len(line) not in [1, 2]:
          raise ValueError(f'Line {line_idx} of vocabulary file, with contents '
                           f'\'{line}\' is malformed')

        idx, word = line if len(line) == 2 else (line_idx, line[0])
        idx = int(idx)

        if idx in idx2word:
          raise ValueError(
              f'Vocabulary contains two words with same index {idx}.')
        if word != word.lower():
          raise ValueError(f'Word {word} with index {idx} is not lower case.')

        idx2word[idx] = word

    # Validate.
    if len(idx2word) != len(set(idx2word.values())):
      raise ValueError('Words in vocabulary are not unique.')
    basic_tokens = {self.PAD, self.BOS, self.EOS, self.UNK}
    if basic_tokens & set(idx2word.values()) != basic_tokens:
      raise ValueError(
          f'Vocabulary does not contain all basic tokens {basic_tokens}.')

    self._idx2word = idx2word
    self._word2idx = {v: k for k, v in idx2word.items()}

    self._vocab_size = len(idx2word)
    self._pad_token = self._word2idx[self.PAD]
    self._bos_token = self._word2idx[self.BOS]
    self._eos_token = self._word2idx[self.EOS]
    self._unk_token = self._word2idx[self.UNK]

    self._tf_word2idx = None
    self._tf_whitespace_tokenizer = None

  def initialize(self):
    ids_tensor = tf.constant([i for w, i in self._word2idx.items()],
                             dtype=tf.int32)
    words_tensor = tf.constant([w for w, i in self._word2idx.items()],
                               dtype=tf.string)
    self._tf_whitespace_tokenizer = tensorflow_text.WhitespaceTokenizer()
    self._tf_word2idx = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(words_tensor, ids_tensor),
        self._unk_token)

  def string_tensor_to_indices(self,
                               string_tensor: Union[tf.Tensor, Sequence[str]],
                               prepend_bos: bool = False,
                               append_eos: bool = False,
                               max_num_tokens: Optional[int] = 32) -> tf.Tensor:
    if self._tf_word2idx is None or self._tf_whitespace_tokenizer is None:
      raise RuntimeError('Model was not initialized. Call `initialize` method.')

    # Remove punctuation.
    string_tensor = tf.strings.regex_replace(string_tensor, '[[:punct:]]', '')
    # Lower case.
    string_tensor = tf.strings.lower(string_tensor)
    if prepend_bos:
      string_tensor = self.BOS.encode('utf-8') + b' ' + string_tensor
    if append_eos:
      string_tensor += b' ' + self.EOS.encode('utf-8')

    # Separate words by whitespace.
    tokenized = self._tf_whitespace_tokenizer.tokenize(string_tensor)
    # Map word to indices.
    tokenized = self._tf_word2idx.lookup(tokenized)
    # Pad to `max_num_tokens`.
    shape = None if max_num_tokens is None else [None, max_num_tokens]
    tokenized = tokenized.to_tensor(default_value=self._pad_token, shape=shape)
    return tokenized

  def indices_to_string(self, indices: Sequence[int]) -> str:
    # Cut at `EOS` or `PAD`.
    idx_list_cut = []
    for token_id in indices:
      if token_id in [self._pad_token, self._eos_token]:
        break
      idx_list_cut.append(token_id)

    # Decode back to string.
    words_list = [self._idx2word[idx] for idx in idx_list_cut]
    return ' '.join(words_list)

  def string_to_indices(self,
                        string: str,
                        prepend_bos: bool = False,
                        append_eos: bool = False,
                        max_num_tokens: Optional[int] = 32) -> Sequence[int]:
    """Tokenizes, mapping a python string to a sequence of indices."""
    string = string.translate(
        str.maketrans('', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    string = string.lower()
    words = string.split(' ')
    tokenized = [self._word2idx.get(w, self._unk_token) for w in words]
    tokenized = [self._bos_token] * prepend_bos + tokenized
    tokenized += [self._eos_token] * append_eos
    if max_num_tokens:
      tokenized = tokenized[:max_num_tokens]
      num_tokens = len(tokenized)
      tokenized = tokenized + [self._pad_token] * (max_num_tokens - num_tokens)
    return tokenized

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def pad_token(self):
    return self._pad_token

  @property
  def bos_token(self):
    return self._bos_token

  @property
  def eos_token(self):
    return self._eos_token

  @property
  def unk_token(self):
    return self._unk_token


class BertTokenizer(TextTokenizer):
  """BERT tokenizer.

  Standard BERT vocabularies can be found in tf hub.
  """

  PAD = '[PAD]'
  CLS = '[CLS]'
  SEP = '[SEP]'
  BOS = CLS
  EOS = SEP
  UNK = '[UNK]'

  def __init__(self, vocabulary_path: str):
    """Initializes the `BertTokenizer`.

    Args:
      vocabulary_path: A path to a vocabulary file. The vocabulary is a simple
        text file where each line is of the form: 'token'. The vocabulary should
        at least contain the words: '[PAD]', '[CLS]', '[SEP]' and '[UNK]'.
    """
    # Parse the vocabulary.
    idx2word = {}
    self._vocabulary_path = vocabulary_path
    with tf.io.gfile.GFile(vocabulary_path) as f:
      for idx, line in enumerate(f):
        word = line.strip()
        idx2word[idx] = word

    # Validate.
    if len(idx2word) != len(set(idx2word.values())):
      raise ValueError('Words in vocabulary are not unique.')
    basic_tokens = {self.PAD, self.BOS, self.EOS, self.UNK}
    if basic_tokens & set(idx2word.values()) != basic_tokens:
      raise ValueError(
          f'Vocabulary does not contain all basic tokens {basic_tokens}.')

    self._idx2word = idx2word
    self._word2idx = {v: k for k, v in idx2word.items()}

    self._vocab_size = len(idx2word)
    self._pad_token = self._word2idx[self.PAD]
    self._bos_token = self._word2idx[self.BOS]
    self._eos_token = self._word2idx[self.EOS]
    self._unk_token = self._word2idx[self.UNK]

    self._tf_tokenizer = None

  def initialize(self):
    self._tf_tokenizer = tensorflow_text.BertTokenizer(
        self._vocabulary_path,
        token_out_type=tf.int32,
        unknown_token=self.UNK,
        lower_case=True)

  def string_tensor_to_indices(self,
                               string_tensor: Union[tf.Tensor, Sequence[str]],
                               prepend_bos: bool = False,
                               append_eos: bool = False,
                               max_num_tokens: Optional[int] = 32) -> tf.Tensor:
    if self._tf_tokenizer is None:
      raise RuntimeError('Model was not initialized. Call `initialize` method.')

    batch_size = tf.shape(input=string_tensor)[0]
    tokenized = self._tf_tokenizer.tokenize(string_tensor)
    tokenized = tokenized.merge_dims(-2, -1)

    if append_eos:
      eos_tensor = tf.ragged.constant([self._eos_token])
      eos_tensor = tf.tile(eos_tensor, [batch_size])
      eos_tensor = tf.expand_dims(eos_tensor, axis=1)
      tokenized = tf.concat([tokenized, eos_tensor], axis=1)
    if prepend_bos:
      bos_tensor = tf.ragged.constant([self._bos_token])
      bos_tensor = tf.tile(bos_tensor, [batch_size])
      bos_tensor = tf.expand_dims(bos_tensor, axis=1)
      tokenized = tf.concat([bos_tensor, tokenized], axis=1)

    # Pad to `max_num_tokens`.
    shape = None if max_num_tokens is None else [None, max_num_tokens]
    tokenized = tokenized.to_tensor(default_value=self._pad_token, shape=shape)
    return tokenized

  def indices_to_string(self, indices: Sequence[int]) -> str:
    # Cut at `EOS` or `PAD`.
    idx_list_cut = []
    for token_id in indices:
      if token_id in [self._pad_token, self._eos_token]:
        break
      idx_list_cut.append(token_id)

    # Decode back to string.
    word_iter = (self._idx2word[idx] for idx in idx_list_cut)
    return ' '.join(word_iter).replace(' ##', '')

  @property
  def vocab_size(self):
    return self._vocab_size

  @property
  def pad_token(self):
    return self._pad_token

  @property
  def bos_token(self):
    return self._bos_token

  @property
  def eos_token(self):
    return self._eos_token

  @property
  def unk_token(self):
    return self._unk_token

  @property
  def cls_token(self):
    return self._bos_token

  @property
  def sep_token(self):
    return self._eos_token


class ClipTokenizer(TextTokenizer):
  """CLIP tokenizer."""

  BOS = '<|startoftext|>'
  EOS = '<|endoftext|>'
  UNK = EOS

  def __init__(
      self,
      vocabulary_path: Optional[str] = None,
  ) -> None:
    """Initializes the `ClipTokenizer`.

    Args:
      vocabulary_path: A path to a CLIP-style vocabulary file.
    """
    self._tokenizer = clip.simple_tokenizer.SimpleTokenizer(vocabulary_path)

    self._vocab_size = len(self._tokenizer.encoder)
    self._pad_token = 0
    self._bos_token = self._tokenizer.encoder[self.BOS]
    self._eos_token = self._tokenizer.encoder[self.EOS]
    self._unk_token = self._tokenizer.encoder[self.UNK]

    self._initialized = False

  def initialize(self) -> None:
    self._initialized = True

  def _clip_tokenize(self, texts: Union[tf.Tensor,
                                        Sequence[str]]) -> tf.RaggedTensor:
    if isinstance(texts, tf.Tensor):
      texts = [text.decode('utf-8') for text in texts._numpy().tolist()]  # pylint: disable=protected-access
    return tf.ragged.constant([self._tokenizer.encode(text) for text in texts],
                              dtype=tf.int32)

  def string_tensor_to_indices(self,
                               string_tensor: Union[tf.Tensor, Sequence[str]],
                               prepend_bos: bool = False,
                               append_eos: bool = False,
                               max_num_tokens: Optional[int] = 77) -> tf.Tensor:
    if not self._initialized:  # To satisfy the tests.
      raise RuntimeError('Model was not initialized. Call `initialize` method.')

    batch_size = tf.shape(input=string_tensor)[0]

    tokenized = tf.py_function(
        func=self._clip_tokenize,
        inp=[string_tensor],
        Tout=tf.RaggedTensorSpec([None, None], dtype=tf.int32))

    if append_eos:
      eos_tensor = tf.ragged.constant([self._eos_token])
      eos_tensor = tf.tile(eos_tensor, [batch_size])
      eos_tensor = tf.expand_dims(eos_tensor, axis=1)
      tokenized = tf.concat([tokenized, eos_tensor], axis=1)
    if prepend_bos:
      bos_tensor = tf.ragged.constant([self._bos_token])
      bos_tensor = tf.tile(bos_tensor, [batch_size])
      bos_tensor = tf.expand_dims(bos_tensor, axis=1)
      tokenized = tf.concat([bos_tensor, tokenized], axis=1)

    # Pad to `max_num_tokens`.
    shape = None if max_num_tokens is None else [None, max_num_tokens]
    return tokenized.to_tensor(default_value=self._pad_token, shape=shape)

  def indices_to_string(self, indices: Sequence[int]) -> str:
    text = self._tokenizer.decode(i for i in indices if i != self._pad_token)
    start_pos = len(self.BOS) if text.startswith(self.BOS) else 0
    end_pos = text.index(self.EOS) if self.EOS in text else None
    return text[start_pos:end_pos].strip()

  @property
  def vocab_size(self) -> int:
    return self._vocab_size

  @property
  def pad_token(self) -> int:
    return self._pad_token

  @property
  def bos_token(self) -> int:
    return self._bos_token

  @property
  def eos_token(self) -> int:
    return self._eos_token

  @property
  def unk_token(self) -> int:
    return self._unk_token
