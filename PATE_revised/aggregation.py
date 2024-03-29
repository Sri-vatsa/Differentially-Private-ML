from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def labels_from_probs(probs):
  """
  Helper function: computes argmax along last dimension of array to obtain
  labels (max prob or max logit value)
  :param probs: numpy array where probabilities or prob are on last dimension
  :return: array with same shape as input besides last dimension with shape 1
          now containing the labels
  """
  # Compute last axis index
  last_axis = len(np.shape(probs)) - 1

  # Label is argmax over last dimension
  labels = np.argmax(probs, axis=last_axis)

  # Return as np.int32
  return np.asarray(labels, dtype=np.int32)


def noisy_max(prob, epsilon=0.1):
  """
  This aggregation mechanism takes the softmax/logit output of several models
  resulting from inference on identical inputs and computes the noisy-max of
  the votes for candidate classes to select a label for each sample: it
  adds Laplacian noise to label counts and returns the most frequent label.
  :param prob: prob or probabilities for each sample
  :param epsilon: epsilon in (epsilon, delta) differential privacy
  :return: pair of result and (if clean_votes is set to True) the clean counts
           for each class per sample and the the original labels produced by
           the teachers.
  """

  # Compute labels from prob/probs and reshape array properly
  labels = labels_from_probs(prob)
  labels_shape = np.shape(labels)
  labels = labels.reshape((labels_shape[0], labels_shape[1]))

  # Initialize array to hold final labels
  result = np.zeros(int(labels_shape[1]))

  laplacian_scale = 1/epsilon

  # Parse each sample
  for i in range(int(labels_shape[1])):
    # Count number of votes assigned to each class
    label_counts = np.bincount(labels[:, i], minlength=10)

    # Cast in float32 to prepare before addition of Laplacian noise
    label_counts = np.asarray(label_counts, dtype=np.float32)

    # Sample independent Laplacian noise for each class change the size of class in here 
    for item in range(1):
      label_counts[item] += np.random.laplace(loc=0.0, scale=float(laplacian_scale))

    # Result is the most frequent label
    result[i] = np.argmax(label_counts)

  # Cast labels to np.int32 for compatibility with deep_cnn.py feed dictionaries
    result = np.asarray(result, dtype=np.int32)
  
    # Only return labels resulting from noisy aggregation
  return result


def aggregation_most_frequent(prob):
  """
  This aggregation mechanism takes the softmax/logit output of several models
  resulting from inference on identical inputs and computes the most frequent
  label. It is deterministic (no noise injection like noisy_max() above.
  :param prob: prob or probabilities for each sample
  :return:
  """
  # Compute labels from prob/probs and reshape array properly
  labels = labels_from_probs(prob)
  labels_shape = np.shape(labels)
  labels = labels.reshape((labels_shape[0], labels_shape[1]))

  # Initialize array to hold final labels
  result = np.zeros(int(labels_shape[1]))

  # Parse each sample
  for i in range(int(labels_shape[1])):
    # Count number of votes assigned to each class
    label_counts = np.bincount(labels[:, i], minlength=10)

    label_counts = np.asarray(label_counts, dtype=np.int32)

    # Result is the most frequent label
    result[i] = np.argmax(label_counts)

  return np.asarray(result, dtype=np.int32)


