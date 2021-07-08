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
r"""Script to generate csvs for HMDB.

You would need to download the official splits at:

http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar

and unrar the archive on your machine, e.g. /path/to/hmdb/

Usage:
```
python generate_hmdb_csv.py  \
  --input_path=/path/to/hmdb/testTrainMulti_7030_splits \
  --output_path=/path/to/hmdb
```
"""

import collections
import csv
import glob
import os

from absl import app
from absl import flags


flags.DEFINE_string(
    'input_path', None, 'Path containing the metadata from HMDB51.')
flags.DEFINE_string(
    'output_path', None, 'Path containing the metadata from HMDB51.')

FLAGS = flags.FLAGS

InputVideo = collections.namedtuple(
    'InputRow',
    ('video_id', 'split', 'subset', 'label_name'))

OutputRow = collections.namedtuple(
    'OutputRow',
    ('video_id', 'start_sec', 'end_sec', 'label_name', 'label_id'))


def main(argv):
  del argv
  all_files = glob.glob(os.path.join(FLAGS.input_path, '*txt'))

  all_rows = []
  label_names = set()

  # Read the files.
  for file_name in all_files:
    base_name = os.path.basename(file_name)
    base_name_split = base_name.split('_')
    label_name = ' '.join(base_name_split[:-2])
    label_name = label_name.replace(' ', '_')
    label_names.add(label_name)
    split = int(base_name[-5])
    with open(file_name, 'r') as f:
      lines = [x.strip().split(' ') for x in f.readlines()]

    for (video_id, ind) in lines:
      if ind == '1':
        all_rows.append(
            InputVideo(video_id, split, 'train', label_name))
      elif ind == '2':
        all_rows.append(
            InputVideo(video_id, split, 'test', label_name))

  # Sort the label names.
  label_names = list(label_names)
  label_names.sort()

  all_csvs = {
      'train_1': [],
      'train_2': [],
      'train_3': [],
      'test_1': [],
      'test_2': [],
      'test_3': [],
  }

  # Generate the csvs rows.
  for row in all_rows:
    csv_name = f'{row.subset}_{row.split}'
    all_csvs[csv_name].append(OutputRow(
        video_id=f'{row.label_name}/{row.video_id}',
        start_sec=0,
        end_sec=20,
        label_name=row.label_name,
        label_id=label_names.index(row.label_name)
        ))

  # Write the csvs.
  for csv_name in all_csvs:
    output_path = os.path.join(FLAGS.output_path, f'{csv_name}.csv')
    print(f'Writing outputs to CSV file {output_path}')
    with open(output_path, 'w') as f:
      writer = csv.writer(f, delimiter=',')
      writer.writerow(
          ['video_path', 'start', 'end', 'label'])

      for row in all_csvs[csv_name]:
        writer.writerow([
            row.video_id, row.start_sec, row.end_sec, row.label_name])

if __name__ == '__main__':
  app.run(main)
