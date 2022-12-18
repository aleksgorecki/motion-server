import tensorflow as tf
import os
import keras_preprocessing.image
import numpy as np


class BitmapModel:

    labels = ["xneg", "xpos", "yneg", "ypos", "zneg", "zpos"]

    @staticmethod
    def get_prototype() -> tf.keras.Sequential:

        model = tf.keras.Sequential(layers=[
            tf.keras.Input(shape=(1, 80, 3)),

            tf.keras.layers.Conv1D(8, kernel_size=(3), strides=(1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),

            tf.keras.layers.Conv1D(16, kernel_size=(3), strides=(1), activation="relu"),
            tf.keras.layers.Dropout(rate=0.2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(len(BitmapModel.labels), activation="softmax")
        ])
        return model


def setup_data_flow(dataset_path: str, val_split: float = 0.2):
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=val_split)

    train_flow = train_generator.flow_from_directory(directory=dataset_path,
                                                    target_size=(1, 80),
                                                    color_mode="rgb",
                                                    class_mode='categorical',
                                                    batch_size=8,
                                                    shuffle=True,
                                                    subset="training"
                                                    )

    val_flow = train_generator.flow_from_directory(directory=dataset_path,
                                                    target_size=(1, 80),
                                                    color_mode="rgb",
                                                    class_mode='categorical',
                                                    batch_size=8,
                                                    shuffle=True,
                                                    subset="training"
                                                    )
    
    return train_flow, val_flow
    

def fit_model(model: tf.keras.Sequential, epochs: int, train_flow, callbacks: list, val_flow):
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

    if not model._is_compiled:
        model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(train_flow, epochs=epochs, validation_data=val_flow, callbacks=callbacks)

def run_prediction(model: tf.keras.Sequential, input_image_path: str, labels: list):

    input_image = keras_preprocessing.image.load_img(input_image_path, color_mode="rgb", target_size=(1, 80))
    input_arr = tf.keras.utils.img_to_array(input_image)
    input_arr = np.expand_dims(input_arr, axis=0)

    predictions = model(input_arr)

    print(predictions)
    
    argmax = np.argmax(predictions[0])

    print(f"Most likely: {labels[argmax]} {predictions[0][argmax]}")

    for idx, pred in enumerate(predictions[0]):
        print(f"{labels[idx]} {pred}")

    result_dict = dict()
    result_dict.update({"class": labels[argmax]})
    result_dict.update({"result": float(predictions[0][argmax])})

    return result_dict


if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    BITMAP_DATASET_PATH = "./dataset_bitmap"

    classes = os.listdir(BITMAP_DATASET_PATH)
    model: tf.keras.Sequential = BitmapModel.get_prototype()
    model.summary()

    train_flow, val_flow = setup_data_flow(BITMAP_DATASET_PATH)

    fit_model(model, 40, train_flow, None, val_flow)

    model.save_weights("./weights_latest_bitmap.h5", overwrite=True, save_format="h5")
