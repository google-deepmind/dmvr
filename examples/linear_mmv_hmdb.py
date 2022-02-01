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

"""HMDB51 linear evaluation of MMV models."""

from absl import app
from absl import flags
from dmvr import builders
import hmdb
import numpy as np
from sklearn import preprocessing
from sklearn import svm
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


flags.DEFINE_enum('model_name', 's3d',
                  ['s3d', 'tsm-resnet50', 'tsm-resnet50x2'],
                  'Which MMV backbone to load.')
flags.DEFINE_string('data_path', '/path/to/hmdb/', 'Path where shards live.')
flags.DEFINE_integer('eval_batch_size', 1,
                     'The batch size for evaluation.')
flags.DEFINE_integer('train_batch_size', 16,
                     'The batch size for training.')
flags.DEFINE_integer('num_train_epochs', 10,
                     'How many epochs to collect features during training.')
flags.DEFINE_integer('num_test_clips', 10,
                     'How many clips to average on during test.')
flags.DEFINE_integer('min_resize', 224,
                     'Min value to resize images to during preprocessing.')
flags.DEFINE_integer('crop_size', 200,
                     'Value to resize images to during preprocessing.')
flags.DEFINE_integer('num_frames', 32, 'Number of video frames.')
flags.DEFINE_integer('stride', 1, 'Stride for video frames.')
flags.DEFINE_integer('hmdb51_split', 1, 'Which split of hmdb51 to use.')


FLAGS = flags.FLAGS

_MODELS2REG = {'s3d': 0.0003,
               'tsm-resnet50': 0.0001,
               'tsm-resnet50x2': 0.0003}


def compute_accuracy_metrics(pred: np.ndarray, gt: np.ndarray,
                             prefix: str = ''):
  order_pred = np.argsort(pred, axis=1)
  assert len(gt.shape) == len(order_pred.shape) == 2
  top1_pred = order_pred[:, -1:]
  top5_pred = order_pred[:, -5:]
  top1_acc = np.mean(top1_pred == gt)
  top5_acc = np.mean(np.max(top5_pred == gt, 1))
  return {prefix + 'top1': top1_acc,
          prefix + 'top5': top5_acc}


def main(argv):
  del argv

  # Load the model.
  sklearn_reg = _MODELS2REG[FLAGS.model_name]
  module = hub.load(f'https://tfhub.dev/deepmind/mmv/{FLAGS.model_name}/1')

  def get_features(input_frames: np.ndarray):
    vision_output = module.signatures['video'](
        tf.constant(tf.cast(input_frames, dtype=tf.float32)))
    return vision_output['before_head'].numpy()

  def collect_features_and_labels(ds: tf.data.Dataset, subset: str):
    """Collect features and labels."""
    features = []
    labels = []
    print(f'Computing features on {subset}')
    examples = iter(tfds.as_numpy(ds))
    num_examples = 0
    for ex in examples:
      vid_representation = get_features(ex[builders.IMAGE_FEATURE_NAME])
      labels.append(ex[builders.LABEL_INDEX_FEATURE_NAME])
      features.append(vid_representation)
      num_examples += ex[builders.LABEL_INDEX_FEATURE_NAME].shape[0]
      if num_examples % 100 == 0:
        print(f'Processed {num_examples} examples.')
    labels = np.concatenate(labels, axis=0)
    features = np.concatenate(features, axis=0)
    print(f'Finish collecting {subset} features of shape {features.shape}')
    return features, labels

  # Generate the training and testing datasets.
  conf_kwargs = dict(
      num_frames=FLAGS.num_frames,
      stride=FLAGS.stride,
      min_resize=FLAGS.min_resize,
      crop_size=FLAGS.crop_size,
      one_hot_label=False)

  train_ds = hmdb.HMDB51Factory(
      FLAGS.data_path, subset='train', split=FLAGS.hmdb51_split).configure(
          is_training=True, **conf_kwargs).make_dataset(
              shuffle=True,
              num_epochs=FLAGS.num_train_epochs,
              batch_size=FLAGS.train_batch_size)

  test_ds = hmdb.HMDB51Factory(
      FLAGS.data_path, subset='test', split=FLAGS.hmdb51_split).configure(
          is_training=False, num_test_clips=FLAGS.num_test_clips,
          **conf_kwargs).make_dataset(shuffle=False,
                                      num_epochs=1,
                                      batch_size=FLAGS.eval_batch_size)

  # Collect features and labels.
  train_features, train_labels = collect_features_and_labels(train_ds, 'train')
  test_features, test_labels = collect_features_and_labels(test_ds, 'test')

  # Train classifier
  print('Training linear classifier!')
  classifier = svm.LinearSVC(C=sklearn_reg)
  scaler = preprocessing.StandardScaler().fit(train_features)
  train_features = scaler.transform(train_features)
  classifier.fit(train_features, train_labels.ravel())
  print('Training done !')

  # Evaluation.
  test_features = scaler.transform(test_features)
  print('Running inference on train')
  pred_train = classifier.decision_function(train_features)
  print('Running inference on test')
  pred_test = classifier.decision_function(test_features)
  if FLAGS.num_test_clips > 1:
    pred_test = np.reshape(
        pred_test, (test_labels.shape[0], -1, pred_test.shape[1]))
    pred_test = pred_test.mean(axis=1)

  # Compute accuracies.
  metrics = compute_accuracy_metrics(pred_train, train_labels, prefix='train_')
  metrics.update(
      compute_accuracy_metrics(pred_test, test_labels, prefix='test_'))
  print(metrics)

if __name__ == '__main__':
  app.run(main)
