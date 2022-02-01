# End-to-end walktrough: from raw data to using the readers for learning

This document walks you through all the steps that go from raw data (a list of
mp4 files), to a format that is compatible with DMVR, to writing a reader to
finally use it in an ML application.


## Requirements

To run the code, you will need to install the following dependencies:

-   python3
-   numpy
-   absl-py
-   pandas
-   Tensorflow
-   [ffmpeg](https://johnvansickle.com/ffmpeg/)
-   [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) [Make sure you pip install ffmpeg-python and not python-ffmpeg]
-   unrar [Only for the HMDB-51 dataset generation example]
-   [scikit-learn](https://scikit-learn.org/) [Only for training linear model on HMDB]

Please make sure the ffmpeg binaries (downloadable
[here](https://johnvansickle.com/ffmpeg/)) are visible from the *PATH*
environment variable and to install its python-ffmpeg python wrapper (and not
ffmpeg-python which is different). Installing python-ffmpeg with pip can be done
in one line with:

```sh
pip install ffmpeg-python
```

## Creating and reading your own DMVR dataset using open-source tools

First, we will describe how to generate your own DMVR dataset as tfrecord files
from your own videos using open-source tools.

Finally, we provide a step-by-step example of how to generate the popular
[HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
action recognition video dataset into the DMVR format.

### Generating your own tfrecord files

#### Creating the input CSV for the generation

To generate a DMVR compatible video dataset using this tool, all you need is to
provide a csv file with the paths of the videos you want to process, together
with additional metadata such as the start/end timestamps, a label or a text
caption. As an example we are going to download two videos (creative common
license):

```sh
wget https://cdn.spacetelescope.org/archives/videos/medium_podcast/heic1608c.mp4 \
-O /tmp/heic1608c.mp4
wget https://upload.wikimedia.org/wikipedia/commons/1/18/BRKM_Javeline_Throw.webm \
-O /tmp/BRKM_Javeline_Throw.webm
```

We can create the following csv with the downloaded videos to process:

```sh
video_path,start,end,label,caption
/tmp/heic1608c.mp4,1.5,6.0,space,the view of the space from a telescope
/tmp/BRKM_Javeline_Throw.webm,0.0,3.0,javeline_throw,someone is throwing a javeline
```

where a more precise description of each column is given below:

| Column name | Description                                        | Optional |
| ----------- | -------------------------------------------------- | -------- |
| video_path  | the path of video to process                       | No       |
| start       | the clip start time (in second)                    | No       |
| end         | the clip end timee (in second)                     | No       |
| label       | A label annotated with the clip (i.e. for          | Yes      |
:             : classification)                                    :          :
| caption     | A free-form text annotated with the clip (i.e. for | Yes      |
:             : captioning or retrieval)                           :          :

Run this following line to create the csv:

```sh
echo -e "video_path,start,end,label,caption\n/tmp/heic1608c.mp4,1.5,3.0,space,hubble\n/tmp/BRKM_Javeline_Throw.webm,0.0,3.0,javeline_throw,someone is throwing a javeline" > /tmp/input.csv
```

#### Generating the tfrecords data using the CSV

Now that we have created a CSV file with the videos we wish to process, we can
generated the tfrecords using the provided code. This can be done by running the
following commands:

```sh
mkdir /tmp/generated_dataset
python generate_from_file.py \
--csv_path=/tmp/input.csv \
--output_path=/tmp/generated_dataset
```

where a description of the arguments is given below:

Arguments    | Description
------------ | -------------------------------------------------------
csv_path     | The path of the input CSV with all the video path.
output_path  | The generated tfrecords output path
num_shards   | The number of tfrecord shards to create (default=1)
decode_audio | Decode and store audio in the tfrecords (default=False)
shuffle_csv  | Whether or not to shuffle the input csv (default=False)

Congratulations! You have created a DMVR compatible dataset from your own file!

### Example: Step-by-step generation of HMDB-51 in the DMVR format

As another example, we provide step-by-step instructions for generating the
[HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
video dataset in the DMVR format.

#### Creating the HMDB-51 input CSV for the generation pipeline

First, you need to download the original splits from the official
[link](http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar).

```sh
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar
```

If you have not
[installed unrar](https://www.tecmint.com/how-to-open-extract-and-create-rar-files-in-linux/)
yet, please install it to extract the rar, and then run:

```sh
unrar x test_train_splits.rar
rm test_train_splits.rar
```

Create the HMDB CSV file using our provided script:

```sh
mkdir hmdb_csv
python generate_hmdb_csv.py \
  --input_path=testTrainMulti_7030_splits \
  --output_path=hmdb_csv
```

This will generate in the *hmdb_csv* folder, 6 csv files: train_1.csv,
test_1.csv, train_2.csv, test_2.csv, train_3.csv and test_3.csv which are the
three train/test splits.

#### Generating the tfrecords data from the generated HMDB-51 CSV

Now that you have generated the HMDB-51 csv, you will need to download and
extract the videos from the official website and store them in a newly created
*hmdb_videos* directory:

```sh
mkdir hmdb_videos
wget https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar \
-P hmdb_videos
cd hmdb_videos
unrar x hmdb51_org.rar
rm hmdb51_org.rar
for video_dir in *rar; do unrar x $video_dir; done
rm *.rar
cd ..
```

You can now run the generation pipeline given any csv split, for example you can
run the generation pipeline on the train set of the first split with the
following command:

```sh
python generate_from_file.py \
  --csv_path=hmdb_csv/train_1.csv \
  --video_root_path=hmdb_videos \
  --output_path=/path/to/hmdb_shards
```

and this will generate the tfrecords in the DMVR format for the HMDB-51 split 1
train set split into *sqrt(num_clips)* shards, where *num_clips* is the number
of video clips from the HMDB-51 split 1 train set.

## Writing a DMVR reader

See `hmdb.py` for an example reader for the data created above.

## Training a linear classifier on top of existing features

The script `linear_mmv_hmdb.py` provides a script evaluating the linear
performance of the recently introduced
[MMV networks](https://arxiv.org/abs/2006.16228) on HMDB51.

To run the script simply do:

```shell
python linear_mmv_hmdb.py \
  --data_path=/path/to/hmdb_shards \
  --model_name=s3d \
  --hmdb51_split=1
```

It supports three different models and the script should reproduce
the following results (as reported in the paper):

Visual Backbone | Results on Linear HMDB51 (avg over 3 splits)
-------  | --------
[S3D-G](https://tfhub.dev/deepmind/mmv/s3d/1) (`s3d`)            | 62.6
[Resnet-50 TSM](https://tfhub.dev/deepmind/mmv/tsm-resnet50/1): (`tsm-resnet50`) | 66.7
[Resnet-50 TSMx2](https://tfhub.dev/deepmind/mmv/tsm-resnet50/1): (`tsm-resnet50x2`)  | 67.1


### References

```bibtex
@inproceedings{alayrac2020self,
  title={{S}elf-{S}upervised {M}ulti{M}odal {V}ersatile {N}etworks},
  author={Alayrac, Jean-Baptiste and Recasens, Adri{\`a} and Schneider, Rosalia and Arandjelovi{\'c}, Relja and Ramapuram, Jason and De Fauw, Jeffrey and Smaira, Lucas and Dieleman, Sander and Zisserman, Andrew},
  booktitle={NeurIPS},
  year={2020}
}
```
