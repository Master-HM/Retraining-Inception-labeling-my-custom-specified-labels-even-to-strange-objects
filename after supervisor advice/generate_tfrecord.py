"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('image_dir', '', 'Path to the image directory')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == "/m/01g317":
        return 1
    elif row_label == "/m/0199g":
        return 2
    elif row_label == "/m/0k4j":
        return 3
    elif row_label == "/m/04_sv":
        return 4
    elif row_label == "/m/05czz6l":
        return 5
    elif row_label == "/m/01bjv":
        return 6
    elif row_label == "/m/07jdr":
        return 7
    elif row_label == "/m/07r04":
        return 8
    elif row_label == "/m/019jd":
        return 9
    elif row_label == "/m/015qff":
        return 10
    elif row_label == "/m/01pns0":
        return 11
    elif row_label == "/m/02pv19":
        return 13
    elif row_label == "/m/015qbp":
        return 14
    elif row_label == "/m/0cvnqh":
        return 15
    elif row_label == "/m/015p6":
        return 16
    elif row_label == "/m/01yrx":
        return 17
    elif row_label == "/m/0bt9lr":
        return 18
    elif row_label == "/m/03k3r":
        return 19
    elif row_label == "/m/07bgp":
        return 20
    elif row_label == "/m/01xq0k1":
        return 21
    elif row_label == "/m/0bwd_0j":
        return 22
    elif row_label == "/m/01dws":
        return 23
    elif row_label == "/m/0898b":
        return 24
    elif row_label == "/m/03bk1":
        return 25
    elif row_label == "/m/01940j":
        return 27
    elif row_label == "/m/0hnnb":
        return 28
    elif row_label == "/m/080hkjn":
        return 31
    elif row_label == "/m/01rkbr":
        return 32
    elif row_label == "/m/01s55n":
        return 33
    elif row_label == "/m/02wmf":
        return 34
    elif row_label == "/m/071p9":
        return 35
    elif row_label == "/m/06__v":
        return 36
    elif row_label == "/m/018xm":
        return 37
    elif row_label == "/m/02zt3":
        return 38
    elif row_label == "/m/03g8mr":
        return 39
    elif row_label == "/m/03grzl":
        return 40
    elif row_label == "/m/06_fw":
        return 41
    elif row_label == "/m/019w40":
        return 42
    elif row_label == "/m/0dv9c":
        return 43
    elif row_label == "/m/04dr76w":
        return 44
    elif row_label == "/m/09tvcd":
        return 46
    elif row_label == "/m/08gqpm":
        return 47
    elif row_label == "/m/0dt3t":
        return 48
    elif row_label == "/m/04ctx":
        return 49
    elif row_label == "/m/0cmx8":
        return 50
    elif row_label == "/m/04kkgm":
        return 51
    elif row_label == "/m/09qck":
        return 52
    elif row_label == "/m/014j1m":
        return 53
    elif row_label == "/m/0l515":
        return 54
    elif row_label == "/m/0cyhj_":
        return 55
    elif row_label == "/m/0hkxq":
        return 56
    elif row_label == "/m/0fj52s":
        return 57
    elif row_label == "/m/01b9xk":
        return 58
    elif row_label == "/m/0663v":
        return 59
    elif row_label == "/m/0jy4k":
        return 60
    elif row_label == "/m/0fszt":
        return 61
    elif row_label == "/m/01mzpv":
        return 62
    elif row_label == "/m/02crq1":
        return 63
    elif row_label == "/m/03fp41":
        return 64
    elif row_label == "/m/03ssj5":
        return 65
    elif row_label == "/m/04bcr3":
        return 67
    elif row_label == "/m/09g1w":
        return 70
    elif row_label == "/m/07c52":
        return 72
    elif row_label == "/m/01c648":
        return 73
    elif row_label == "/m/020lf":
        return 74
    elif row_label == "/m/0qjjc":
        return 75
    elif row_label == "/m/01m2v":
        return 76
    elif row_label == "/m/050k8":
        return 77
    elif row_label == "/m/0fx9l":
        return 78
    elif row_label == "/m/029bxz":
        return 79
    elif row_label == "/m/01k6s3":
        return 80
    elif row_label == "/m/0130jx":
        return 81
    elif row_label == "/m/040b_t":
        return 82
    elif row_label == "/m/0bt_c3":
        return 84
    elif row_label == "/m/01x3z":
        return 85
    elif row_label == "/m/02s195":
        return 86
    elif row_label == "/m/01lsmm":
        return 87
    elif row_label == "/m/0kmg4":
        return 88
    elif row_label == "/m/03wvsk":
        return 89
    elif row_label == "/m/012xff":
        return 90
    elif row_label == "cotton":
        return 91
    elif row_label == "other":
        return 92
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(os.getcwd(), FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()
