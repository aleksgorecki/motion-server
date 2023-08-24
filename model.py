import tensorflow as tf
import os
import keras_preprocessing.image
import numpy as np
from PIL import ImageFont

from data_processing import MotionDataset
from motion import Motion
from typing import Dict, List


MOTION_LEN = 120
CLASSES = ["nothing", "x_negative", "x_positive", "y_negative", "y_positive", "z_negative", "z_positive"]
DATASET_PATH_H5 = "./basic_dataset.hdf5"
DATASET_VAL_PATH_H5 = "./basic_dataset_val.hdf5"
MODEL_SAVE_PATH = "./model_latest.h5"
EPOCHS = 100

TFLITE_SAVE_PATH = "./model.tflite"


def get_prototype_model(motion_len: int = MOTION_LEN, num_classes: int = len(CLASSES)) -> tf.keras.Sequential:
    model = tf.keras.Sequential(layers=[
        tf.keras.Input(shape=(1, motion_len, 3)),

        tf.keras.layers.Conv1D(16, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Conv1D(32, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model


def get_no_dropout_model(motion_len: int = MOTION_LEN, num_classes: int = len(CLASSES)):
        model = tf.keras.Sequential(layers=[
            tf.keras.Input(shape=(1, motion_len, 3)),

            tf.keras.layers.Conv1D(16, kernel_size=(3), strides=(1), activation="relu"),

            tf.keras.layers.Conv1D(32, kernel_size=(3), strides=(1), activation="relu"),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation="softmax")
        ])
        return model


def get_3layer_model(motion_len: int = MOTION_LEN, num_classes: int = len(CLASSES)) -> tf.keras.Sequential:
    model = tf.keras.Sequential(layers=[
        tf.keras.Input(shape=(1, motion_len, 3)),

        tf.keras.layers.Conv1D(16, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Conv1D(32, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Conv1D(64, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model


def get_smaller_model(motion_len: int = MOTION_LEN, num_classes: int = len(CLASSES)) -> tf.keras.Sequential:
    model = tf.keras.Sequential(layers=[
        tf.keras.Input(shape=(1, motion_len, 3)),

        tf.keras.layers.Conv1D(8, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Conv1D(16, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.4),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model


def run_prediction(model: tf.keras.Sequential, motion: Motion, labels=None) -> Dict["str", "str"]:

    if labels is None:
        labels = CLASSES

    input_arr = np.expand_dims(motion.get_array(), axis=0)
    predictions = model(input_arr)

    argmax = np.argmax(predictions[0])
    print(argmax)
    print(f"Most likely: {labels[argmax]} {predictions[0][argmax]}")
    for idx, pred in enumerate(predictions[0]):
        print(f"{labels[idx]} {pred}")

    result_dict = dict()
    result_dict.update({"class": labels[argmax]})
    result_dict.update({"result": str(float(predictions[0][argmax]))})

    return result_dict


def convert_to_tflite(source_model_path: str, new_model_path: str) -> None:
    model = tf.keras.models.load_model(source_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(new_model_path, 'wb') as f:
        f.write(tflite_model)
