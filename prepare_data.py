import tensorflow as tf
import json
import numpy as np
from tensorflow.keras import layers, models
class Meta:
    def __init__(self, value):
        self.categories = set()
        self.type = type(value)
        if isinstance(value, list):
            self.min = min(value)
            self.max = max(value)
            self.categories.update(value)
        elif self.type == dict:
            self.min = min(value.keys())
            self.max = max(value.keys())
            self.categories.update(value.keys())
        else:
            self.min = value
            self.max = value
            if self.type in [int, str]:
                self.categories.add(value)

    def Update(self, value):
        value_type = type(value)
        if self.type == int and value_type == float:
            self.type = float
        if isinstance(value, list):
            self.min = min(value + [self.min])
            self.max = max(value + [self.max])
        elif self.type == dict:
            self.min = min(list(value.keys()) + [self.min])
            self.max = max(list(value.keys()) + [self.max])
        else:
            if self.min > value:
                self.min = value 
            if self.max < value:
                self.max = value
            if self.type in [int, str]:
                self.categories.add(value)
    def Export(self):
        self.type = self.type.__name__
        self.categories = sorted(self.categories)
        return self.__dict__

def read_feature(filenames):
    for filename in filenames:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                uid, features_json, dt = line.rstrip().split('\t')
                yield json.loads(features_json)

def make_tfrecord(in_files=['train.tsv'], out_file='train.tfrecord', out_meta='train.meta'):
    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    metadata = dict()
    with tf.io.TFRecordWriter(out_file) as writer:
        for features_dict in read_feature(in_files):
            tf_features = dict()
            for key, value in features_dict.items():
                if value is None:
                    continue
                # append key value
                value_type = type(value)
                if value_type in [list, dict] and len(value) == 0:
                    continue
                if value_type == int:
                    tf_features[key] = _int64_feature([value])
                elif value_type == float:
                    tf_features[key] = _float_feature([value])
                elif value_type == str:
                    tf_features[key] = _bytes_feature([value.encode()])

                # append list:
                elif isinstance(value, list):
                    if type(value[0]) == int:
                        tf_features[key] = _int64_feature(value)
                    if type(value[0]) == str:
                        tf_features[key] = _bytes_feature([s.encode() for s in value])
                elif value_type == dict:
                    tf_features[key+'_keys'] = _bytes_feature([s.encode() for s in value.keys()])
                    tf_features[key+'_values'] = _float_feature([float(v) for v in value.values()])

                # update metadata
                if key not in metadata:
                    metadata[key] = Meta(value)
                else:
                    metadata[key].Update(value)

            example_proto = tf.train.Example(features=tf.train.Features(feature=tf_features))
            serialized_example = example_proto.SerializeToString() # tf.train.Example.FromString(serialized_example)
            writer.write(serialized_example)
    with open(out_meta, 'w') as f:
        json.dump({key:value.Export() for key, value in metadata.items()}, f, indent=2)

def make_columns(meta_file = 'train.meta'):
    with open(meta_file, 'r') as f:
        metadata = json.load(f)
    columns = []
    for key, meta in metadata.items():
        if meta['type'] == 'float':
            column = tf.feature_column.numeric_column(key, dtype=tf.float32, default_value=0)
        elif meta['type'] == 'str':
            column = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    key, meta['categories']
                )
            )
        elif meta['type'] == 'list':
            column = tf.feature_column.indicator_column(
                tf.feature_column.categorical_column_with_vocabulary_list(
                    key, meta['categories']
                )
            )
            """
            elif meta['type'] == 'int':
                column = tf.feature_column.indicator_column(
                    tf.feature_column.categorical_column_with_vocabulary_list(
                        key, meta['categories']
                    )
                )
            """
        else:
            continue
        if True: #key in ['gender', 'gender_prob', 'age', 'subscribe_channel_list']: #'gender', 'gender_prob', 
            columns.append(column)
    return columns

feature_description = {
    'gender': tf.io.FixedLenFeature([], tf.string), #, default_value='unknown'
    'gender_prob': tf.io.FixedLenFeature([], tf.float32, default_value=0),
    'age': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    #'age_seg_18_24': tf.io.FixedLenFeature([], tf.float32, default_value=0),
    #'city_code': tf.io.FixedLenFeature([], tf.string, default_value='unknown'),
    #'prefecture_id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    #'device_type': tf.io.FixedLenFeature([], tf.string, default_value='unknown'),
    #'subscribe_channel_list': tf.io.FixedLenFeature([], tf.string, default_value='unknown')
    'subscribe_channel_list': tf.io.VarLenFeature(tf.string)
}

if __name__ == '__main__':
    make_tfrecord()
    columns = make_columns()
    spec = tf.feature_column.make_parse_example_spec(columns)
    print (repr(spec))
    def _parse_function_with_column(example_proto):
        return tf.io.parse_single_example(example_proto, spec) # parse_single_example
    def _parse_function_with_desc(example_proto):
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        return parsed

    raw_dataset = tf.data.TFRecordDataset(['train.tfrecord'])

    parsed_dataset = raw_dataset.map(_parse_function_with_column)
    batch_dataset = parsed_dataset.batch(10)
    for batch in batch_dataset.take(1):
        print (repr(batch))
        dense = layers.DenseFeatures(columns)
        feature_dense = dense(batch).numpy()
        print (feature_dense)
