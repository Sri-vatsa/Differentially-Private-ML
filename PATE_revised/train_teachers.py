from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

import partition
import models

def train_teacher (nb_teachers, teacher_id):
  """
  This function trains a single teacher model with responds teacher's ID among an ensemble of nb_teachers
  models for the dataset specified.
  The model will be save in directory. 
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # Load the dataset
  X_train, X_test, y_train, y_test = models.get_dataset()

  print(X_train.shape)
  print(y_train.shape)
  print(X_test.shape)
  print(y_test.shape)
  
  # Retrieve subset of data for this teacher
  data, labels = partition.partition_dataset(X_train,
                                         y_train,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path

  filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.hdf5'
  filename2 = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.h5'
 
  # Perform teacher training need to modify 
 

  # Create teacher model
  model, opt = models.create_two_layer_mlp(46) # num of cols
  model.compile(loss='binary_crossentropy',
              optimizer="Adam",
              metrics=['accuracy'])
  model, hist = models.training(model, data, X_test, labels, y_test,filename)

  #modify
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
  model.save_weights(filename2)
  print("Saved model to disk")
  return True


num_teachers = 10
for i in range(num_teachers):
  train_teacher(num_teachers, i)


