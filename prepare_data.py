import tensorflow as tf
import json
import numpy as np
from tensorflow.keras import layers, models
class Meta:
    def __init__(self, value):
        self.category = set()
        self.type = type(value)
        if isinstance(value, list):
            self.min = min(value)
            self.max = max(value)
            self.category.update(value)
        elif self.type == dict:
            self.min = min(value.keys())
            self.max = max(value.keys())
            self.category.update(value.keys())
        else:
            self.min = value
            self.max = value
            if self.type in [int, str]:
                self.category.add(value)

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
                self.category.add(value)
    def Export(self):
        self.type = self.type.__name__
        self.category = sorted(self.category)
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


make_tfrecord()

raw_dataset = tf.data.TFRecordDataset(['train.tfrecord'])
feature_description = {
    'gender': tf.io.FixedLenFeature([], tf.string, default_value='unknown'),
    'gender_prob': tf.io.FixedLenFeature([], tf.float32, default_value=0),
    'age': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'age_seg_18_24': tf.io.FixedLenFeature([], tf.float32, default_value=0),
    'city_code': tf.io.FixedLenFeature([], tf.string, default_value='unknown'),
    'prefecture_id': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'device_type': tf.io.FixedLenFeature([], tf.string, default_value='unknown'),
}

def make_columns():
    age = tf.feature_column.numeric_column('age', dtype=tf.int64, default_value=0)
    gender = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_vocabulary_list(
            'gender', ['male', 'female'], 
        )
    )
    return [age, gender]
columns = make_columns()
spec = tf.feature_column.make_parse_example_spec(columns)
print (repr(spec))
def _parse_function_with_column(example_proto):
    return tf.io.parse_single_example(example_proto, spec)
"""
#dense_tensor = input_layer(features, columns)
feature_layer = tf.keras.layers.DenseFeatures(columns)
print (feature_layer(data).numpy())
"""
def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

def test1():
    parsed_dataset = raw_dataset.map(_parse_function)
    #parsed_dataset
    for parsed_record in parsed_dataset.take(10):
        print (repr(parsed_record))


parsed_dataset = raw_dataset.map(_parse_function_with_column)
for parsed_record in parsed_dataset.take(10):
    print (repr(parsed_record))
    dense = layers.DenseFeatures(columns)
    feature_dense = dense(parsed_record).numpy()
    print (feature_dense)