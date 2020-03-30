import tensorflow as tf
import json
import numpy as np
from tensorflow.keras import layers, models
from collections import defaultdict

class Meta:
    def __init__(self, value):
        self.categories = defaultdict(int)
        self.count = 0
        self.type = type(value)
        if self.type == list:
            self.min = min(value)
            self.max = max(value)
            self.UpdateCategories(value)
        elif self.type == dict:
            self.min = min(value.keys())
            self.max = max(value.keys())
            self.UpdateCategories(value.keys())
        else:
            self.min = value
            self.max = value
            if self.type in [int, str]:
                self.UpdateCategories([value])

    def Update(self, value):
        value_type = type(value)
        if value_type == list:
            self.min = min(value + [self.min])
            self.max = max(value + [self.max])
            self.UpdateCategories(value)
        elif self.type == dict:
            self.min = min(list(value.keys()) + [self.min])
            self.max = max(list(value.keys()) + [self.max])
            self.UpdateCategories(value.keys())
        else:
            if self.type == int and value_type == float:
                self.type = float
            if self.min > value:
                self.min = value
            if self.max < value:
                self.max = value
            if self.type in [int, str]:
                self.UpdateCategories([value])

    def UpdateCategories(self, cats):
        for cat in cats:
            self.categories[cat] += 1

    def Export(self):
        self.type = self.type.__name__
        self.categories = sorted(self.categories)
        return self.__dict__


def read_features(filenames, target='ctr'):
    for filename in filenames:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                ots, uid, campaign_id, label, features, _type, dt = line.rstrip().split('\t')
                if (target == _type):
                    features_dict = json.loads(features)
                    features_dict['label'] = float(label)
                    yield features_dict


def make_tfrecord(in_files, out_files, out_meta):
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            # BytesList won't unpack a string from an EagerTensor.
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=list(value)))
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

    metadata = dict()
    for in_file, out_file in zip(in_files, out_files):
        with tf.io.TFRecordWriter(out_file) as writer:
            for features_dict in read_features([in_file]):
                tf_features = dict()
                for key, value in features_dict.items():
                    # Handle blank value.
                    if value is None:
                        continue
                    value_type = type(value)
                    if value_type in [list, dict] and len(value) == 0:
                        continue
                    
                    # Handle auf_* features. These features are string of float.
                    if key.startswith('auf_') and value_type == str:
                        value = float(value)
                        value_type = float
                    # Handle age feature. This feature is int. We need float here.
                    if key == 'age':
                        value = float(value)
                        value_type = float

                    # Append scalar feature.
                    if value_type == int:
                        tf_features[key] = _int64_feature([value])
                    elif value_type == float:
                        tf_features[key] = _float_feature([value])
                    elif value_type == str:
                        tf_features[key] = _bytes_feature([value.encode()])
                    # Append list.
                    elif isinstance(value, list):
                        if type(value[0]) == int:
                            tf_features[key] = _int64_feature(value)
                        if type(value[0]) == str:
                            tf_features[key] = _bytes_feature(
                                [s.encode() for s in value])
                    # Append map.
                    elif value_type == dict:
                        tf_features[key+'_keys'] = _bytes_feature(
                            [s.encode() for s in value.keys()])
                        tf_features[key+'_values'] = _float_feature(
                            [float(v) for v in value.values()])

                    # Update metadata.
                    if key not in metadata:
                        metadata[key] = Meta(value)
                    else:
                        metadata[key].Update(value)

                example_proto = tf.train.Example(
                    features=tf.train.Features(feature=tf_features))
                serialized_example = example_proto.SerializeToString()
                # to unseriealize: tf.train.Example.FromString(serialized_example)
                writer.write(serialized_example)
    with open(out_meta, 'w') as f:
        json.dump({key: value.Export()
                   for key, value in metadata.items()}, f, indent=2)


def make_columns(meta_file):
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    columns = []
    columns_name = [name for name in sorted(metadata.keys()) if name != 'label']
    for key in columns_name:
        meta = metadata[key]
        # skip sparse features
        if len(meta['categories']) > 100:
            print (key)
            continue
        if meta['type'] == 'float':
            column = tf.feature_column.numeric_column(
                key, dtype=tf.float32, default_value=0)
        elif meta['type'] in ['str', 'int', 'list']:
            column = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    key, meta['categories']))
        elif meta['type'] == 'dict':
            categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(
                key+'_keys', meta['categories'])
            weighted_column = tf.feature_column.weighted_categorical_column(
                categorical_column, key+'_values')
            column = tf.feature_column.indicator_column(weighted_column)
        else:
            continue
        columns.append(column)
    return columns

if __name__ == '__main__':
    prefix = 'train_v3'
    make_tfrecord([prefix+'.tsv'], [prefix+'.tfrecord'], prefix+'.meta')
    
    columns = make_columns(prefix+'.meta')
    spec = tf.feature_column.make_parse_example_spec(columns)
    print(repr(spec))

    def _parse_function_with_column(example_proto, desc=spec):
        # parse_single_example
        return tf.io.parse_single_example(example_proto, desc)

    raw_dataset = tf.data.TFRecordDataset([prefix + '.tfrecord'])

    parsed_dataset = raw_dataset.map(_parse_function_with_column)
    batch_dataset = parsed_dataset.batch(10)

    for batch in batch_dataset.take(1):
        print(repr(batch))
        dense = layers.DenseFeatures(columns)
        feature_dense = dense(batch).numpy()
        print(feature_dense.shape)
        print(feature_dense)
    print(sorted(c.name for c in columns))
