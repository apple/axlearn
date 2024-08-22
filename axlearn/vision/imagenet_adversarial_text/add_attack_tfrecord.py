"""Add text to imagenet images, and save a new tfrecord.

Run on a Macbook:
    python add_attack_tfrecord.py \
        --dir_input=/path/to/dataset/imagenet2012/ \
        --dir_output=/path/to/dataset/imagenet2012_attack/
"""

import json
import os
from glob import glob
from io import BytesIO

import tensorflow as tf
from absl import app, flags

# TODO(llcao): change the abstract path to relative path
from axlearn.vision.imagenet_adversarial_text import util_im_process, util_imagenet, util_tfdata

sc = util_imagenet.ImageNet_SimilarClass()

flags.DEFINE_string("dir_input", None, "input folder of tfrecord", required=True)
flags.DEFINE_string("dir_output", None, "output folder.", required=True)
flags.DEFINE_bool("single_file_for_debug", False, "whether to process single file for debugging.")

FLAGS = flags.FLAGS

imagenet_classes = [""] * 1000
with open("imagenet-simple.json", encoding="utf8") as f:
    imagenet_classes_dict = json.load(f)
    for c in range(1000):
        imagenet_classes[c] = imagenet_classes_dict[str(c)]


def _decode_record_and_attack(record: dict[str, tf.Tensor], verbose: bool = False):
    """Decodes a record to a TensorFlow example.

    We first use tf.io.decode_image(example['image']).numpy()
    and then use BytesIO to write a PIL Image to in-memory jpg file (bytes)

    Args:
       record: one example in tfrecord file (a tensor or serialized string)
       verbose: bool value controlling whether to show debug info.

    Note tfrecord does not correlate PIL, and the following does not work
    if 0: # TypeError img_tensor is not byte
      new_img_array = np.uint8(new_img)
      new_img_bytes = tf.convert_to_tensor(new_img_array)

    if 0: # PIL Image tobytes() does not get an image format
      new_img_bytes = new_img.tobytes()

    """
    example = tf.io.parse_single_example(
        serialized=record,
        features={
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "file_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    target = example["label"].numpy()
    img = tf.io.decode_image(example["image"]).numpy()
    img = util_im_process.to_rgb(img)

    if verbose:
        print(f"image shape: {img.shape}")

    new_category = sc.most_similar_class(target)

    new_text = imagenet_classes[new_category]
    new_img = util_im_process.write_text_to_image(img, new_text)

    temp = BytesIO()
    new_img.save(temp, format="jpeg")
    new_img_bytes = temp.getvalue()

    my_features = {
        "image": util_tfdata.bytes_feature(new_img_bytes),
        "label": util_tfdata.int64_feature(target),
        "file_name": util_tfdata.bytes_feature(example["file_name"].numpy()),
    }
    new_example = tf.train.Example(features=tf.train.Features(feature=my_features))
    return new_example


def process_one_tf_record(fn_tfrecord_input: str, fn_tfrecord_output: str):
    """Process one tf record, and save attacked examples to another tfrecord.

    Args:
         fn_tfrecord_input: path of input file
         fn_tfrecord_output: path of output file

    To parse/save tfrecord, we use tf.train.Feature,  tf.train.Example and
      tf.io.parse_single_example.

    Example of reading
    example = tf.io.parse_single_example(serialized=record, features={
         "image": tf.io.FixedLenFeature([], tf.string),
         "label": tf.io.FixedLenFeature([], tf.int64),
         "file_name": tf.io.FixedLenFeature([], tf.string),
     })

    Example of saving
    new_example = tf.train.Example(features=tf.train.Features(feature=my_features = {
      'my_key1': tf.train.Feature(bytes_list=tf.train.BytesList(value=[value1])),
      'my_key2': tf.train.Feature(float_list=tf.train.FloatList(value=[value2])),
     }))
    """

    ds = tf.data.TFRecordDataset(fn_tfrecord_input)

    with tf.io.TFRecordWriter(fn_tfrecord_output) as tfwriter:
        # TODO(llcao): use tf map function to speedup.
        # Refer to:
        # https://stackoverflow.com/questions/61720708/how-do-you-save-a-tensorflow-dataset-to-a-file
        for record in ds:
            new_example = _decode_record_and_attack(record)
            tfwriter.write(new_example.SerializeToString())


def main(_):
    if FLAGS.single_file_for_debug:
        fn0 = "imagenet2012-validation.tfrecord-00000-of-00064"
        fns_tfrecords = [FLAGS.dir_input + fn0]
    else:
        fns_tfrecords = glob(FLAGS.dir_input + "*-validation.tfrecord-*")

    for ii, fn_input in enumerate(fns_tfrecords):
        fn_output = os.path.basename(fn_input)
        fn_output = FLAGS.dir_output + fn_output
        print(f"{ii} out of {len(fns_tfrecords)}")
        process_one_tf_record(fn_input, fn_output)


if __name__ == "__main__":
    app.run(main)
