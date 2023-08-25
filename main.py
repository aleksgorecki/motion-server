from flask import Flask, request
import time
import json
import model
from motion import Motion
import os
import tensorflow as tf


app = Flask(__name__)
prediction_model = tf.keras.models.load_model(model.MODEL_SAVE_PATH)


@app.route("/")
def check_server():
    return "OK"


@app.route("/new", methods=["POST"])
def new_recording():
    data = request.json

    motion_class = data["class"]
    dataset_dir = data["dataset"]
    class_dir = os.path.join(dataset_dir, motion_class)
    os.makedirs(class_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    motion_data_file = f"{motion_class}_{timestamp}.json"
    path_to_save = os.path.join(class_dir, motion_data_file)

    with open(path_to_save, "w") as file:
        json.dump(data, file, indent=4)

    return "OK"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    received_sample = Motion.from_json(data)
    received_sample.crop(
        received_sample.get_global_extremum_position(), model.MOTION_LEN // 2
    )
    prediction_result = model.run_prediction(
        prediction_model, received_sample, model.CLASSES
    )

    return app.response_class(
        response=json.dumps(prediction_result), status=200, mimetype="application/json"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
