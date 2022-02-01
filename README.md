# DMVR: DeepMind Video Readers

DMVR is a library providing a framework for easily reading raw data and
producing `tf.data.Dataset` objects ready to be consumed by models.

## Design principles

### Data processing graph

The main idea of the framework is to build a customizable and reusable data
processing graph that when applied to raw data files, will produce final dataset
objects. Building blocks called Builders are used to interact with the graph by
adding, removing or replacing data processing blocks.

Dataset providers can write a Factory with a default data processing graph for
each dataset. Dataset consumers can customize the graph to their needs either by
creating a child Factory or just appending a given instance. Factory objects
expose instances of Builders allowing control of the multiple phases of the data
processing graph. The Factory is then able to generate `tf.data.Dataset`
objects.

### Phases

The data processing graph is split in multipple phases. This abstraction is
purely semantic, which makes code easier to reuse. The phases are:

-   Parse
-   Sample
-   Decode
-   Preprocess
-   Postprocess

### Modalities

In order to easily add different modalities to the dataset from the raw data,
sub graphs for some modalities with default processing (e.g. sample, decode and
crop for images) is provided. These sub graphs can be added by simply calling
the corresponding methods for the Builders.

## Usage

### Dataset providers

Dataset providers should implement a factory populating the default graph.

Example:

-   Data is stored in TFRecords as `tf.train.SequenceExample` objects.

```python
from typing import List

from dmvr import modalities
from dmvr import video_dataset

class Kinetics700Factory(video_dataset.BaseVideoDatasetFactory):

  _NUM_CLASSES = 700

  def __init__(self, subset: str):
    self._is_training = subset == 'train'
    shards: List[str] = path_to_the_data(subset)
    super().__init__(shards)

  def _build(self,
             # Video related parameters.
             num_frames: int = 32,
             stride: int = 1,
             num_test_clips: int = 1,
             min_resize: int = 224,
             crop_size: int = 200,
             zero_centering_image: bool = False,
             # Label related parameters.
             one_hot_label: bool = True,
             add_label_name: bool = False):
    """Build default data processing graph."""
    modalities.add_image(parser_builder=self.parser_builder,
                         sampler_builder=self.sampler_builder,
                         decoder_builder=self.decoder_builder,
                         preprocessor_builder=self.preprocessor_builder,
                         postprocessor_builder=self.postprocessor_builder,
                         is_training=self._is_training,
                         num_frames=num_frames,
                         stride=stride,
                         min_resize=min_resize,
                         crop_size=crop_size,
                         zero_centering_image=zero_centering_image)

    modalities.add_label(parser_builder=self.parser_builder,
                         decoder_builder=self.decoder_builder,
                         preprocessor_builder=self.preprocessor_builder,
                         one_hot_label=one_hot_label,
                         num_classes=self._NUM_CLASSES,
                         add_label_name=add_label_name)
```

### Dataset consumers

Dataset consumers can create `tf.data.Dataset` objects from a factory instance.

Example:

```python
factory = Kinetics700Factory('train')
factory.configure(num_frames=16)
ds = factory.make_dataset(batch_size=8)
```

The user can also customize the data processing graph by adding more functions:

```python
from dmvr import builders
from dmvr import processors

factory = Kinetics700Factory('train')
factory.configure(num_frames=16)

factory.preprocess_builder.add_fn(processors.scale_jitter_augm,
                                  feature_name=builders.IMAGE_FEATURE_NAME)
factory.preprocess_builder.add_fn(processors.color_default_augm,
                                  feature_name=builders.IMAGE_FEATURE_NAME)

ds = factory.make_dataset(batch_size=8)
```

## Installation

DMVR can be installed with pip directly from github, with the following command:

pip install git+git://github.com/deepmind/dmvr.git

Python 3.9+ is required in order for all features to be available.
