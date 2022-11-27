from flask import Flask, request
from matplotlib import pyplot as plt
import os
import pathlib
import json
import time


DATASET_DIR = "dataset"

app = Flask(__name__)


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


if __name__ == "__main__":

    if os.path.exists(DATASET_DIR):
        if (pathlib.Path(DATASET_DIR)).is_file():
            raise Exception("Dataset dir is a file")
        
    app.run(host="0.0.0.0", port=8080)
