import sys
import os
# Adding parent folder to path so we can import from our own project
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import argparse
import functools
import glob

import tensorflow as tf
import numpy as np
import PIL.Image
from PIL import ImageChops

import MachineLearning.inception_resnet_v1 as inception_resnet_v1

def model_fn(features, labels, mode, params):
  images = features['image']
  num_classes = params.get('num_classes', 2)
  checkpoint_path = params.get('checkpoint_path', None)
  checkpoint_ignore_vars = params.get('checkpoint_ignore_vars', '')
  train_variable_names = params.get('train_vars', '')
  model_dir = params.get('model_dir', 'model_dir')

  is_training = mode == tf.estimator.ModeKeys.TRAIN

  keep_prob = 0.7 if mode == tf.estimator.ModeKeys.TRAIN else 1.0

  logits, end_points = inception_resnet_v1.inception_resnet_v1(
      images,
      dropout_keep_prob=keep_prob,
      num_classes=num_classes,
      is_training=is_training)

  eval_metric_ops = None
  train_op = None
  loss = None
  predictions = end_points

  if checkpoint_path:
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    if not latest_checkpoint:
      ignore_vars = []
      if checkpoint_ignore_vars:
        ignore_vars = checkpoint_ignore_vars.split(',')
      restore_dict = {}
      reader = tf.train.NewCheckpointReader(checkpoint_path)
      for var in tf.trainable_variables():
        tensor_name = var.name.split(':')[0]
        if reader.has_tensor(tensor_name):
          if tensor_name not in ignore_vars:
            restore_dict[tensor_name] = var

      tf.train.init_from_checkpoint(checkpoint_path, restore_dict)

  train_variables = None
  if train_variable_names:
    train_variables = [v for v in tf.trainable_variables()
        if v.name.split(':')[0] in train_variable_names]

  if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
    onehot_labels = tf.one_hot(labels, num_classes, dtype=tf.float32)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    loss = tf.Print(loss, [loss], message='Loss')
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
          onehot_labels, predictions['Predictions']),
        'class-accuracy': tf.metrics.accuracy(
          labels=labels, predictions=tf.argmax(input=logits, axis=1)),
        }
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss,
        var_list=train_variables,
        global_step=tf.train.get_global_step())

  spec = tf.estimator.EstimatorSpec(
      mode,
      loss=loss,
      train_op=train_op,
      predictions=predictions,
      eval_metric_ops=eval_metric_ops,
      )

  return spec

def pad_image(size, image):
  # from https://stackoverflow.com/a/9103783/8860642
  image.thumbnail(size, PIL.Image.ANTIALIAS)
  image_size = image.size

  thumb = image.crop((0, 0, size[0], size[1]))

  offset_x = max((size[0] - image_size[0]) / 2, 0)
  offset_y = max((size[1] - image_size[1]) / 2, 0)

  thumb = ImageChops.offset(thumb, offset_x, offset_y)

  return np.array(thumb)

def dataset_from_image_files(image_files, image_size, batch_size):
  images = []
  for img in image_files:
    img = PIL.Image.open(img)
    img = pad_image(image_size, img)
    img = np.array(img, dtype=np.float32)
    images.append(img)
  images = np.array(images)
  images = np.divide(images, 255.0, dtype=np.float32)

  images = tf.data.Dataset.from_tensor_slices(images)
  images = images.batch(batch_size)

  iterator = images.make_one_shot_iterator()
  n = iterator.get_next()
  return {'image': n}

def label_decoder(label):
  return label

def image_decoder(image, image_format):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.divide(image, 255)

  return image

def image_preprocess(image, image_size):
  img_dims = tf.shape(image)
  img_height = tf.cast(img_dims[0], tf.float32)
  img_width = tf.cast(img_dims[1], tf.float32)

  random_crop = tf.random_uniform([], minval=0.0, maxval=0.1)
  offset_height = tf.cast(img_height * random_crop, tf.int32)
  offset_width = tf.cast(img_width * random_crop, tf.int32)
  crop_height = tf.cast(img_height - img_height * random_crop, tf.int32)
  crop_width = tf.cast(img_width - img_width * random_crop, tf.int32)

  image = tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  image = tf.image.resize_images(image, image_size)
  # These random operations are copy pasted from
  # https://github.com/tensorflow/models/blob/5a5d330539dff11eef79ca2e716fb477baf13cf9/research/slim/preprocessing/inception_preprocessing.py#L66
  image = tf.image.random_brightness(image, 32.0 / 255.0)
  image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
  image = tf.image.random_hue(image, max_delta=0.2)
  #image = tf.image.random_contrast(image, lower=0.5, upper=1.5)

  return image

def img_batch_random_rotate(images):
  image_batch_size = tf.shape(images)[0]
  random_angles = tf.random_uniform([image_batch_size], minval=-0.3, maxval=0.3)
  images = tf.contrib.image.rotate(images, random_angles)

  return images

def input_fn(dataset_files, image_size, batch_size, suffle_buffer_size, repeat=True):

  dataset = tf.data.TFRecordDataset(dataset_files)

  def parser(record):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string),
        'image/format': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/class/label': tf.FixedLenFeature((), tf.int64),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = image_decoder(parsed['image/encoded'], parsed['image/format'])
    label = label_decoder(parsed['image/class/label'])

    image = image_preprocess(image, image_size)

    return {'image': image}, label

  dataset = dataset.map(parser)
  dataset = dataset.batch(batch_size)
  if repeat:
    dataset = dataset.repeat()
  if suffle_buffer_size:
    dataset = dataset.shuffle(buffer_size=suffle_buffer_size)

  dataset = dataset.prefetch(10)

  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()

  images = features['image']
  images = img_batch_random_rotate(images)

  return {'image': images}, labels

def main(args):

  tf.logging.set_verbosity(tf.logging.INFO)

  num_classes = args.num_classes

  # Create run config for the estimator
  conditional = {}
  if args.save_checkpoints_secs:
    conditional['save_checkpoints_secs'] = args.save_checkpoints_secs
  else:
    conditional['save_checkpoints_steps'] = args.save_checkpoints_steps
  run_config = tf.estimator.RunConfig().replace(
      model_dir=args.model_dir,
      save_summary_steps=args.save_summary_steps,
      log_step_count_steps=args.log_step_count_steps,
      keep_checkpoint_every_n_hours=args.keep_checkpoint_every_n_hours,
      **conditional
      )
  params = {
      'model_dir': args.model_dir,
      'learning_rate': args.learning_rate,
      'num_classes': num_classes,
      'checkpoint_path': args.checkpoint_path,
      'checkpoint_ignore_vars': args.checkpoint_ignore_vars,
      'train_vars': args.train_vars,
      }

  # Create the estimator
  estimator = tf.estimator.Estimator(
      model_fn=model_fn, config=run_config, params=params)

  image_size = (224, 224)

  if args.action == 'train':
    batch_size = args.batch_size
    suffle_buffer_size = args.suffle_buffer_size
    eval_steps = args.eval_steps

    train_dataset_files = None
    if args.train_dataset_glob:
      train_dataset_files = glob.glob(args.train_dataset_glob)
    else:
      if not args.train_dataset_files:
        assert 'train_dataset_glob or train_dataset_files must be provided'
      train_dataset_files = args.train_dataset_files

    eval_dataset_files = None
    if args.eval_dataset_glob:
      eval_dataset_files = glob.glob(args.eval_dataset_glob)
    else:
      if not args.eval_dataset_files:
        assert 'eval_dataset_glob or eval_dataset_files must be provided'
      eval_dataset_files = args.eval_dataset_files


    train_input_fn = functools.partial(input_fn,
        train_dataset_files, image_size, batch_size, suffle_buffer_size)
    eval_input_fn = functools.partial(input_fn,
        eval_dataset_files, image_size, batch_size, suffle_buffer_size)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn, max_steps=args.max_train_steps)
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn, steps=args.eval_steps, throttle_secs=args.eval_throttle_secs)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  elif args.action == 'predict':
    image_files = args.images
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    fn = functools.partial(dataset_from_image_files, image_files, image_size, batch_size)
    predictions = []
    for p in estimator.predict(fn, predict_keys=['Predictions'], checkpoint_path=checkpoint_path):
      predictions.append(p)

    for i, p in zip(image_files, predictions):
      print('{}: {}'.format(i, p))

