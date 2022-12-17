from __future__ import annotations
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import json
from numpy.typing import NDArray, ArrayLike
from motion import Motion


def plot_motion(motion: Motion, savefile: str):
    fig, ax = plt.subplots()
    axis_names = ["x", "y", "z"]
    for idx, motion_axis in enumerate((motion.get_x(), motion.get_y(), motion.get_z())):
        ax.plot(range(len(motion)), motion_axis, label=axis_names[idx])
    ax.legend()
    ax.set_ylabel("acceleration m/s^2")
    ax.set_xlabel("sample number")
    fig.savefig(savefile)


def dataset_to_bitmap(dataset_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    class_dirs = os.listdir(dataset_path)
    for class_dir in class_dirs:
        os.makedirs(os.path.join(output_path, class_dir), exist_ok=True)
        samples_jsons = os.listdir(os.path.join(dataset_path, class_dir))
        for sample_json in samples_jsons:
            with open(os.path.join(dataset_path, class_dir, sample_json), "r") as json_file_handle:
                json_object = json.load(json_file_handle)
            motion_sample: Motion = Motion.from_json(json_object)
            motion_sample.crop(motion_sample.get_global_extremum_position(), 40)
            output_sample_path = os.path.join(output_path, class_dir, str(sample_json).replace("json", "bmp"))
            motion_sample.save_as_bitmap(output_sample_path)


if __name__ == "__main__":
    #dataset_to_bitmap("./dataset", "./dataset_bitmap")
    with open("./dataset/zneg/zneg_20221204_182105.json", "r") as json_file_handle:
        json_object = json.load(json_file_handle)
    data: Motion = Motion.from_json(json_object)
    x = data.samples[0][0][:]
    data.crop(data.get_global_extremum_position(), 40)
    plot_motion(data, "test_plot.jpg")
    data.horizontal_shift(-0.2)
    data.average_filter(5)
    plot_motion(data, "test_plot_shifted.jpg")
