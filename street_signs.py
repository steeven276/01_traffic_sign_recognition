import os
import glob
import sklearn
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from my_utils import split_data, order_test_set, create_generators
from deep_learning_model import streetsigns_model

if __name__=="__main__":
    if False:
        path_to_data="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\Train"
        path_to_save_train="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\training_data\\train"
        path_to_save_val="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\training_data\\val"

        split_data(path_to_data=path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)
    if False:
        path_to_images="C:\\Users\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\Test"
        path_to_csv="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\Test.csv"
        order_test_set(path_to_images, path_to_csv)

    path_to_train="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\training_data\\train"
    path_to_val="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\training_data\\val"
    path_to_test="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\Test"

    batch_size=64
    epochs=15

    train_generator, val_generator, test_generator=create_generators(batch_size=batch_size, train_data_path=path_to_train, val_data_path=path_to_val, test_data_path=path_to_test)
    nbr_classes=train_generator.num_classes


    TRAIN=False
    TEST=True

    if TRAIN:
        path_to_save_model='./Models'

        ckpt_saver=ModelCheckpoint(
            path_to_save_model,
            monitor="val_accuracy",
            mode='max',
            save_best_only=True,
            save_freq='epoch',
            verbose=1
        )


        early_stop=EarlyStopping(
            monitor="val_accuracy",
            patience=10
        )


        model=streetsigns_model(nbr_classes)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics='accuracy'
        )
        model.fit(
            train_generator,
            epochs= epochs,
            batch_size=batch_size,
            validation_data=val_generator,
            callbacks=[ckpt_saver,early_stop]
        )

    if TEST:
        model=tf.keras.models.load_model('./Models')
        model.summary()

        print("Evaluating validation set:")
        model.evaluate(val_generator)
        print("Evaluating test set :")
        model.evaluate(test_generator)


