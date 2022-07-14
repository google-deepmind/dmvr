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

"""Tests for tokenizers."""

from __future__ import annotations

from collections.abc import Sequence
import os
from typing import Type, TypeVar

import clip.simple_tokenizer
from dmvr import tokenizers
from parameterized import parameterized
import tensorflow as tf

# Removed: Internal pyglib dependencies

_TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'testdata')
_MOCK_DATA = os.path.join(_TESTDATA_DIR, 'tokenizers')

_FILENAMES = {
    tokenizers.SentencePieceTokenizer: 'spiece.model.1000.model',
    tokenizers.WordTokenizer: 'word_vocab.txt',
    tokenizers.BertTokenizer: 'bert_word_vocab.txt',
    tokenizers.ClipTokenizer: clip.simple_tokenizer.default_bpe(),
}

T = TypeVar('T', bound=tokenizers.TextTokenizer)


def _get_tokenizer(cls: Type[T]) -> T:
  filename = _FILENAMES[cls]
  path = os.path.join(_MOCK_DATA, filename)  # OSS: removed internal filename loading.
  return cls(path)


def _tokenize_with_original_clip(
    texts: str | Sequence[str],
    context_length: int = 77) -> Sequence[Sequence[int]]:
  # Code adapted from `clip.tokenize` because it's not importable (only
  # `clip.simple_tokenizer` is).

  if isinstance(texts, str):
    texts = [texts]

  tokenizer = clip.simple_tokenizer.SimpleTokenizer()
  sot_token = tokenizer.encoder['<|startoftext|>']
  eot_token = tokenizer.encoder['<|endoftext|>']
  all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token]
                for text in texts]
  result = []

  for i, tokens in enumerate(all_tokens):
    if len(tokens) > context_length:
      raise RuntimeError(f'Input {texts[i]} is too long for context length'
                         f' {context_length}')
    result.append(tokens + [0] * (context_length - len(tokens)))

  return result


def _decode_with_original_clip(tokens_ids: Sequence[int]) -> str:
  tokenizer = clip.simple_tokenizer.SimpleTokenizer()
  text = tokenizer.decode(tokens_ids)

  eos = '<|endoftext|>'
  return text[:text.index(eos) + len(eos)]


class TokenizerTest(tf.test.TestCase):

  @parameterized.expand(
      ((tokenizers.WordTokenizer,), (tokenizers.SentencePieceTokenizer,),
       (tokenizers.BertTokenizer,), (tokenizers.ClipTokenizer,)))
  def test_tokenizer(self, cls):
    tokenizer = _get_tokenizer(cls)
    tokenizer.initialize()
    input_string = ['hello world']

    tokenized = tokenizer.string_tensor_to_indices(
        input_string, max_num_tokens=42)
    self.assertEqual(tokenized.dtype, tf.int32)

    tokenized = tokenized.numpy().tolist()[0]
    self.assertLen(tokenized, 42)
    self.assertEqual(tokenized[-1], tokenizer.pad_token)

    detokenized = tokenizer.indices_to_string(tokenized)
    self.assertEqual(detokenized, 'hello world')

  @parameterized.expand(
      ((tokenizers.WordTokenizer,), (tokenizers.SentencePieceTokenizer,),
       (tokenizers.BertTokenizer,), (tokenizers.ClipTokenizer,)))
  def test_bos_eos(self, cls):
    tokenizer = _get_tokenizer(cls)
    tokenizer.initialize()
    input_string = ['hello world']

    tokenized = tokenizer.string_tensor_to_indices(
        input_string, prepend_bos=True, append_eos=True)
    tokenized = tokenized.numpy().tolist()[0]
    self.assertEqual(tokenized[0], tokenizer.bos_token)

    if tokenizer.pad_token != tokenizer.eos_token:
      tokenized = [t for t in tokenized if t != tokenizer.pad_token]
    self.assertEqual(tokenized[-1], tokenizer.eos_token)

  @parameterized.expand(
      ((tokenizers.WordTokenizer,), (tokenizers.SentencePieceTokenizer,),
       (tokenizers.BertTokenizer,), (tokenizers.ClipTokenizer,)))
  def test_not_initialized(self, cls):
    tokenizer = _get_tokenizer(cls)
    input_string = ['hello world']

    with self.assertRaises(RuntimeError):
      tokenizer.string_tensor_to_indices(input_string)

  @parameterized.expand((
      (tokenizers.WordTokenizer,),
      (tokenizers.SentencePieceTokenizer,),
  ))
  def test_string_to_indices(self, cls):
    tokenizer = _get_tokenizer(cls)
    tokenizer.initialize()
    input_string = 'hello world'
    tokenized = tokenizer.string_to_indices(
        input_string, prepend_bos=True, append_eos=True, max_num_tokens=42)
    self.assertEqual(type(tokenized), list)
    self.assertEqual(tokenized[0], tokenizer.bos_token)
    tokenized = [t for t in tokenized if t != tokenizer.pad_token]
    self.assertEqual(tokenized[-1], tokenizer.eos_token)

    detokenized = tokenizer.indices_to_string(tokenized[1:-1])
    self.assertEqual(detokenized, 'hello world')

  def test_clip_tokenizer(self):
    tokenizer = _get_tokenizer(tokenizers.ClipTokenizer)
    tokenizer.initialize()
    input_string = ['This is a test.', 'pushups']
    actual_tokenized_tf = tokenizer.string_tensor_to_indices(
        input_string, prepend_bos=True, append_eos=True, max_num_tokens=77)

    expected_tokenized = _tokenize_with_original_clip(input_string)

    actual_tokenized1 = actual_tokenized_tf.numpy().tolist()[0]
    expected_tokenized1 = expected_tokenized[0]
    self.assertEqual(actual_tokenized1, expected_tokenized1)

    actual_decoded = tokenizer.indices_to_string(actual_tokenized1)
    self.assertEqual(actual_decoded, 'this is a test .')

    actual_tokenized2 = actual_tokenized_tf.numpy().tolist()[1]
    expected_tokenized2 = expected_tokenized[1]
    self.assertEqual(actual_tokenized2, expected_tokenized2)

    actual_decoded = tokenizer.indices_to_string(actual_tokenized2)
    self.assertEqual(actual_decoded, input_string[1])


if __name__ == '__main__':
  tf.test.main()
