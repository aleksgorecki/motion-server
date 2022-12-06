from flask import Flask, request
from matplotlib import pyplot as plt
import os
import pathlib
import json
import time
from model import *
from data_processing import *




app = Flask(__name__)

DATASET_DIR = "dataset"
temp_filename = "./received_bitmap_latest.bmp"

model = BitmapModel.get_prototype()
model.load_weights("./weights_latest_bitmap.h5")


@app.route("/")
def check_server():
    return "OK"


@app.route("/new", methods=["POST"])
def new_recording():
    data = request.json

    motion_class = data["class"]

    class_dir = os.path.join(DATASET_DIR, motion_class)
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
    data_arr = json_data_to_array(data)
    normalized_arr = normalize_amplitude_values(data_arr)
    # filtered_arr = filter_high_frequency_oscillations(normalized_arr, 15)

    save_array_to_bitmap(normalized_arr, temp_filename)

    print('\n')
    run_prediction(model, temp_filename, BitmapModel.labels)
    return "OK"


if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        if (pathlib.Path(DATASET_DIR)).is_file():
            raise Exception("Dataset dir is a file")
        
    app.run(host="0.0.0.0", port=8080)
