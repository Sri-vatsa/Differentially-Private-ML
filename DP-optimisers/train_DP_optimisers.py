from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import privacy.optimizers.dp_optimizer as optimizers

tf.flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.flags.DEFINE_integer('epochs', 15, 'Number of epochs')

FLAGS = tf.flags.FLAGS


def cnn_model_fn(features, labels, mode):
  """Model function for a CNN."""

  # Define CNN architecture using tf.keras.layers.
  input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
  y = tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu').apply(input_layer)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu').apply(y)
  y = tf.keras.layers.MaxPool2D(2, 1).apply(y)
  y = tf.keras.layers.Flatten().apply(y)
  y = tf.keras.layers.Dense(32, activation='relu').apply(y)
  logits = tf.keras.layers.Dense(10).apply(y)

  # Calculate loss as a vector and as its average across minibatch.
  vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                               logits=logits)
  scalar_loss = tf.reduce_mean(vector_loss)

  # Configure the training op (for TRAIN mode).
  if mode == tf.estimator.ModeKeys.TRAIN:
    # optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    opt_loss = scalar_loss
    global_step = tf.train.get_global_step()
    # train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)

    learning_rate = 0.25
    noise_multiplier = 1.2
    l2_norm_clip = 1.5
    batch_size = 256
    epochs = 15
    num_microbatches = 16

    optimizer = optimizers.DPGradientDescentGaussianOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss=vector_loss)

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      train_op=train_op)

  # Add evaluation metrics (for EVAL mode).
  elif mode == tf.estimator.ModeKeys.EVAL:
    eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(
                labels=labels,
                predictions=tf.argmax(input=logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=scalar_loss,
                                      eval_metric_ops=eval_metric_ops)


def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  train, test = tf.keras.datasets.mnist.load_data()
  train_data, train_labels = train
  test_data, test_labels = test

  train_data = np.array(train_data, dtype=np.float32) / 255
  test_data = np.array(test_data, dtype=np.float32) / 255

  train_labels = np.array(train_labels, dtype=np.int32)
  test_labels = np.array(test_labels, dtype=np.int32)

  assert train_data.min() == 0.
  assert train_data.max() == 1.
  assert test_data.min() == 0.
  assert test_data.max() == 1.
  assert train_labels.ndim == 1
  assert test_labels.ndim == 1

  return train_data, train_labels, test_data, test_labels


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Load training and test data.
  train_data, train_labels, test_data, test_labels = load_mnist()

  # Instantiate the tf.Estimator.
  mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn)

  # Create tf.Estimator input functions for the training and test data.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.epochs,
      shuffle=True)
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=False)

  # Training loop.
  steps_per_epoch = 60000 // FLAGS.batch_size
  for epoch in range(1, FLAGS.epochs + 1):
    # Train the model for one epoch.
    mnist_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)

    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    test_accuracy = eval_results['accuracy']
    print('Test accuracy after %d epochs is: %.3f' % (epoch, test_accuracy))

if __name__ == '__main__':
  tf.app.run()
