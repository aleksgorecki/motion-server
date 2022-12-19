import tensorflow as tf
import os
import keras_preprocessing.image
import numpy as np
from data_processing import MotionDataset
from motion import Motion
from typing import Dict, List


MOTION_LEN = 120
CLASSES = ["nothing", "x_negative", "x_positive", "y_negative", "y_positive", "z_negative", "z_positive"]
DATASET_PATH_H5 = "./basic_dataset.hdf5"
DATASET_VAL_PATH_H5 = "./basic_dataset_val.hdf5"
MODEL_SAVE_PATH = "./model_latest.h5"
EPOCHS = 200


def get_prototype_model(motion_len: int = MOTION_LEN, num_classes: int = len(CLASSES)) -> tf.keras.Sequential:
    model = tf.keras.Sequential(layers=[
        tf.keras.Input(shape=(1, motion_len, 3)),

        tf.keras.layers.Conv1D(8, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.2),

        tf.keras.layers.Conv1D(16, kernel_size=(3), strides=(1), activation="relu"),
        tf.keras.layers.Dropout(rate=0.2),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model


def run_prediction(model: tf.keras.Sequential, motion: Motion, labels=None) -> Dict["str", "str"]:

    if labels is None:
        labels = CLASSES

    input_arr = np.expand_dims(motion.get_array(), axis=0)
    predictions = model(input_arr)

    argmax = np.argmax(predictions[0])
    print(f"Most likely: {labels[argmax]} {predictions[0][argmax]}")
    for idx, pred in enumerate(predictions[0]):
        print(f"{labels[idx]} {pred}")

    result_dict = dict()
    result_dict.update({"class": labels[argmax]})
    result_dict.update({"result": str(float(predictions[0][argmax]))})

    return result_dict


if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model: tf.keras.Sequential = get_prototype_model()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    dataset = MotionDataset.from_hdf5(DATASET_PATH_H5)
    val_dataset = MotionDataset.from_hdf5(DATASET_VAL_PATH_H5)

    dataset.even_sample_numbers_in_classes()
    val_dataset.even_sample_numbers_in_classes()

    dataset = dataset.to_tf_dataset()
    val_dataset = val_dataset.to_tf_dataset()

    dataset = dataset.batch(8)
    val_dataset = val_dataset.batch(8)

    model.fit(x=dataset, epochs=EPOCHS, validation_data=val_dataset, shuffle=True)
    model.save(MODEL_SAVE_PATH, save_format="h5")
