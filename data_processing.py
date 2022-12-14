from __future__ import annotations
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import json
from numpy.typing import ArrayLike


class Motion:
    x: ArrayLike
    y: ArrayLike
    z: ArrayLike
    duration_ms: int
    class_label: str

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, duration_ms: int, class_label=None) -> None:
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.duration_ms = duration_ms
        self.class_label = class_label

    def from_json(json_object) -> Motion:
        return Motion(
            x=json_object['x'], 
            y=json_object['y'], 
            z=json_object['z'], 
            duration_ms=json_object['duration_ms'], 
            class_label=json_object.get('class')
            )

    def get_global_extremum_sample(self) -> int:
        
        possible_extremas = dict()
        
        for axis_vals in (self.x, self.y, self.z):
            axis_minimum_arg = np.argmin(axis_vals)
            axis_maximum_arg = np.argmax(axis_vals)

            axis_minimum = axis_vals[axis_minimum_arg]
            axis_maximum = axis_vals[axis_maximum_arg]

            if axis_maximum > abs(axis_minimum):
                possible_extremas.update( {axis_maximum_arg: axis_maximum} )
            else:
                possible_extremas.update( {axis_minimum_arg: abs(axis_minimum)} )

        global_extremum_sample = max(possible_extremas, key=possible_extremas.get)
        
        return global_extremum_sample

    def crop(self, center_position: int, half_span: int) -> None:
        lower_bound = center_position - half_span
        if lower_bound < 0:
            lower_bound = 0
        
        full_axis_len = len(self.x)
        upper_bound = center_position + half_span
        if upper_bound > full_axis_len:
            upper_bound = full_axis_len

        self.x = self.x[lower_bound : upper_bound]
        self.y = self.y[lower_bound : upper_bound]
        self.z = self.z[lower_bound : upper_bound]

    def scale_down_values(self, factor: int) -> None:
        self.x = self.x / factor
        self.y = self.y / factor
        self.z = self.z / factor

    def filter_high_frequencies(self) -> None:
        ones_len = int(len(self.x) / 10)
        self.x = np.convolve(self.x, np.ones(ones_len), 'same') / ones_len
        self.y = np.convolve(self.y, np.ones(ones_len), 'same') / ones_len
        self.z = np.convolve(self.z, np.ones(ones_len), 'same') / ones_len

    def save_to_bitmap(self, savefile: str) -> None:
        im_array = np.empty(shape=(1, len(self.x), 3))

        for sample in range(len(self.x)):
            im_array[0][sample][0] = self.x[sample]
            im_array[0][sample][1] = self.y[sample]
            im_array[0][sample][2] = self.z[sample]

        im_min = im_array.min()
        im_max = im_array.max()

        im_array = ((im_array - im_min) / (im_max - im_min)) * 255

        scaled_arr = np.array(im_array, dtype=np.uint8)
        im = Image.fromarray(scaled_arr, mode="RGB")
        im.save(savefile, format="BMP")

    
def plot_motion(motion: Motion, savefile: str):
    fig, ax = plt.subplots()
    axis_names = ["x", "y", "z"]
    for idx, motion_axis in enumerate((motion.x, motion.y, motion.z)):
        ax.plot(range(len(motion_axis)), motion_axis, label=axis_names[idx])
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
            motion_sample.crop(motion_sample.get_global_extremum_sample(), 40)
            motion_sample.filter_high_frequencies()
            output_sample_path = os.path.join(output_path, class_dir, str(sample_json).replace("json", "bmp") )
            motion_sample.save_to_bitmap(output_sample_path)


if __name__ == "__main__":
    dataset_to_bitmap("./dataset", "./dataset_bitmap")
    # with open("./dataset/xpos/xpos_20221204_171220.json", "r") as json_file_handle:
    #     json_object = json.load(json_file_handle)
    # data: Motion = Motion.from_json(json_object)
    # data.crop(data.get_global_extremum_sample(), 40)
    # plot_motion(data, "test_plot.jpg")