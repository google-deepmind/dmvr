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

import os
from typing import Type, TypeVar

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
}

T = TypeVar('T', bound=tokenizers.TextTokenizer)


def _get_tokenizer(cls: Type[T]) -> T:
  filename = _FILENAMES[cls]
  path = os.path.join(_MOCK_DATA, filename)  # OSS: removed internal filename loading.
  return cls(path)


class TokenizerTest(tf.test.TestCase):

  @parameterized.expand(
      ((tokenizers.WordTokenizer,), (tokenizers.SentencePieceTokenizer,),
       (tokenizers.BertTokenizer,)))
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
       (tokenizers.BertTokenizer,)))
  def test_bos_eos(self, cls):
    tokenizer = _get_tokenizer(cls)
    tokenizer.initialize()
    input_string = ['hello world']

    tokenized = tokenizer.string_tensor_to_indices(
        input_string, prepend_bos=True, append_eos=True)
    tokenized = tokenized.numpy().tolist()[0]
    self.assertEqual(tokenized[0], tokenizer.bos_token)
    tokenized = [t for t in tokenized if t != tokenizer.pad_token]
    self.assertEqual(tokenized[-1], tokenizer.eos_token)

  @parameterized.expand(
      ((tokenizers.WordTokenizer,), (tokenizers.SentencePieceTokenizer,),
       (tokenizers.BertTokenizer,)))
  def test_not_initialized(self, cls):
    tokenizer = _get_tokenizer(cls)
    input_string = ['hello world']

    with self.assertRaises(RuntimeError) as _:
      tokenizer.string_tensor_to_indices(input_string)

  @parameterized.expand(
      ((tokenizers.WordTokenizer,), (tokenizers.SentencePieceTokenizer,),))
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


if __name__ == '__main__':
  tf.test.main()
