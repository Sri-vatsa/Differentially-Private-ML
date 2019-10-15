import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py
import pandas as pd
import glob
import time
from random import shuffle
from collections import Counter

from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import model_from_json


#hyperparameters
map_characters ={0:'no cancer' , 1:'cancer'}
pic_size = 140
batch_size = 200
epochs = 20
num_classes = len(map_characters)
test_size = 0.15

def get_dataset():
    df_full = pd.read_csv('../data/UCI_cervical_cancer.csv')
    df_fullna = df_full.replace('?', np.nan)
    df = df_fullna.apply(pd.to_numeric)

    df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
    df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
    df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
    df['Smokes'] = df['Smokes'].fillna(1)
    df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
    df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
    df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
    df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
    df['IUD'] = df['IUD'].fillna(0) # Under suggestion
    df['IUD (years)'] = df['IUD (years)'].fillna(0) #Under suggestion
    df['STDs'] = df['STDs'].fillna(1)
    df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
    df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())
    df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(df['STDs:cervical condylomatosis'].median())
    df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(df['STDs:vaginal condylomatosis'].median())
    df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(df['STDs:vulvo-perineal condylomatosis'].median())
    df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
    df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(df['STDs:pelvic inflammatory disease'].median())
    df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())
    df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(df['STDs:molluscum contagiosum'].median())
    df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
    df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
    df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
    df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())
    df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].median())
    df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].median())

    df = pd.get_dummies(data=df, columns=['Smokes','Hormonal Contraceptives','IUD','STDs',
                                      'Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Hinselmann','Citology','Schiller'])
    
    np.random.seed(42)
    df_data_shuffle = df.iloc[np.random.permutation(len(df))]

    df_train = df_data_shuffle.iloc[1:686, :]
    df_test = df_data_shuffle.iloc[686: , :]


    #features/labels
    df_train_feature = df_train[['Age', 'Number of sexual partners', 'First sexual intercourse',
        'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
        'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
        'STDs:condylomatosis', 'STDs:cervical condylomatosis',
        'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
        'STDs:syphilis', 'STDs:pelvic inflammatory disease',
        'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
        'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
        'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 
        'Smokes_0.0', 'Smokes_1.0',
        'Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0',
        'IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1',
        'Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1',
        'Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1','Schiller_0','Schiller_1']]

    train_label = np.array(df_train['Biopsy'])

    df_test_feature = df_test[['Age', 'Number of sexual partners', 'First sexual intercourse',
        'Num of pregnancies', 'Smokes (years)', 'Smokes (packs/year)',
        'Hormonal Contraceptives (years)', 'IUD (years)', 'STDs (number)',
        'STDs:condylomatosis', 'STDs:cervical condylomatosis',
        'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis',
        'STDs:syphilis', 'STDs:pelvic inflammatory disease',
        'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
        'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
        'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis', 
        'Smokes_0.0', 'Smokes_1.0',
        'Hormonal Contraceptives_0.0', 'Hormonal Contraceptives_1.0', 'IUD_0.0',
        'IUD_1.0', 'STDs_0.0', 'STDs_1.0', 'Dx:Cancer_0', 'Dx:Cancer_1',
        'Dx:CIN_0', 'Dx:CIN_1', 'Dx:HPV_0', 'Dx:HPV_1', 'Dx_0', 'Dx_1',
        'Hinselmann_0', 'Hinselmann_1', 'Citology_0', 'Citology_1','Schiller_0','Schiller_1']]

    test_label = np.array(df_test['Biopsy'])


    #Normalization
    from sklearn import preprocessing
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_feature = minmax_scale.fit_transform(df_train_feature)
    test_feature = minmax_scale.fit_transform(df_test_feature)

    return train_feature, test_feature, train_label, test_label


def create_two_layer_mlp(input_dim):
    model = Sequential() 

    #Input layer
    model.add(Dense(units=500, 
                    input_dim=input_dim, 
                    kernel_initializer='uniform', 
                    activation='relu'))
    model.add(Dropout(0.5))

    #Hidden layer 1
    model.add(Dense(units=200,  
                    kernel_initializer='uniform', 
                    activation='relu'))
    model.add(Dropout(0.5))

    #Output layer
    model.add(Dense(units=1,
                    kernel_initializer='uniform', 
                    activation='sigmoid'))

    opt = "Adam"

    return model, opt

def lr_schedule(epoch):
    #used for the learning rate decay. 
    lr = 0.01
    return lr*(0.1**int(epoch/10))

def load_model_from_checkpoint(weights_path, two_mlp=True, input_dim=46):
    
    if two_mlp:
        model, opt = create_two_layer_mlp(input_dim)
    else:
        model, opt = create_two_layer_mlp(input_dim)
    model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    return model


def training(model, X_train, X_test, y_train, y_test,filepath, data_augmentation=False):
    """
    Training.
    :param model: Keras sequential model
    :param data_augmentation: boolean for using the pre image-processing from keras or not(default:True)
    :param callback: boolean for saving model checkpoints and get the best saved mode
    :return: model and epochs history (acc, loss, val_acc, val_loss for every epochs)
    """
    if data_augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).


        datagen.fit(X_train)
        
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
        callbacks_list = [LearningRateScheduler(lr_schedule) ,checkpoint]
        history = model.fit_generator(datagen.flow(X_train, y_train,
                                    batch_size=batch_size),
                                    steps_per_epoch=X_train.shape[0] // batch_size,
                                    epochs=epochs,
                                    validation_data=(X_test, y_test),
                                    callbacks=callbacks_list)


                        
    else:
        history = model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)
    return model, history


