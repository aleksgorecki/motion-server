import tensorflow as tf
import os
import keras_preprocessing.image


class BitmapModel:

    labels = ["xneg", "xpos", "yneg", "ypos", "zneg", "zpos"]

    @staticmethod
    def get_prototype() -> tf.keras.Sequential:

        model = tf.keras.Sequential(layers=[
            tf.keras.Input((3, 150, 1)),

            tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPool2D((2, 2), padding="same"),

            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPool2D((2, 2), padding="same"),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation="relu", padding="same"),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.MaxPool2D((2, 2), padding="same"),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Dense(len(BitmapModel.labels), activation="softmax")
        ])
        return model


def setup_data_flow(dataset_path: str, val_split: float = 0.2):
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=val_split)

    train_flow = train_generator.flow_from_directory(directory=dataset_path,
                                                    target_size=(3, 150),
                                                    color_mode="rgba",
                                                    class_mode='categorical',
                                                    batch_size=8,
                                                    shuffle=True,
                                                    subset="training"
                                                    )

    val_flow = train_generator.flow_from_directory(directory=dataset_path,
                                                    target_size=(3, 150),
                                                    color_mode="rgba",
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

def run_prediction(model: tf.keras.Sequential, weights_path: str, input_image: str, labels: list):

    model.load_weights(weights_path)
    
    keras_preprocessing.image.load_img(input_image, color_mode="grayscale", target_size=(3, 150))
    predictions = model.predict(input_image)

    for idx, pred in enumerate(predictions):
        print(f"{labels[idx]} {pred}")


if __name__ == "__main__":

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    BITMAP_DATASET_PATH = "./dataset_bitmap"

    classes = os.listdir(BITMAP_DATASET_PATH)
    model: tf.keras.Sequential = BitmapModel.get_prototype()
    model.summary()

    train_flow, val_flow = setup_data_flow(BITMAP_DATASET_PATH)

    fit_model(model, 10, train_flow, None, val_flow)

    model.save_weights("./weights_latest_bitmap.h5", overwrite=True, save_format="h5")
