from __future__ import annotations
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import json
from numpy.typing import NDArray, ArrayLike


class Motion:
    samples: NDArray  # samples[1][len_of_motion][3]

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike, three_channel_array: NDArray = None) -> None:
        if three_channel_array is None:
            self.samples = np.empty(shape=(1, len(x), 3))
            for sample in range(len(x)):
                self.samples[0][sample][0] = x[sample]
                self.samples[0][sample][1] = y[sample]
                self.samples[0][sample][2] = z[sample]
        else:
            self.samples = three_channel_array

    @staticmethod
    def from_json(json_object) -> Motion:
        return Motion(
            x=json_object['x'],
            y=json_object['y'],
            z=json_object['z']
        )

    @staticmethod
    def from_array(array: NDArray) -> Motion:
        return Motion(
            three_channel_array=array
        )

    def __len__(self):
        return self.samples.shape[1]

    def get_separated_axis(self, axis_idx: int):
        axis_arr = np.empty(shape=(len(self)))
        for idx, sample in enumerate(self.samples[0]):
            axis_arr[idx] = sample[axis_idx]
        return axis_arr

    def get_x(self) -> NDArray:
        return self.get_separated_axis(0)

    def get_y(self) -> NDArray:
        return self.get_separated_axis(1)

    def get_z(self) -> NDArray:
        return self.get_separated_axis(2)

    def get_global_extremum_position(self) -> int:

        max_pos = np.unravel_index(self.samples.argmax(), shape=self.samples.shape)
        max_val = self.samples[max_pos]
        min_pos =  np.unravel_index(self.samples.argmin(), shape=self.samples.shape)
        min_val = self.samples[min_pos]

        if max_val > abs(min_val):
            return max_pos[1]
        else:
            return min_pos[1]

    def crop(self, center_position: int, half_span: int) -> None:
        lower_bound = center_position - half_span
        lower_padding = 0
        if lower_bound < 0:
            lower_padding = abs(lower_bound)
            lower_bound = 0

        full_axis_len = self.samples.shape[1]
        upper_bound = center_position + half_span
        upper_padding = 0
        if upper_bound > full_axis_len:
            upper_padding = upper_bound - full_axis_len
            upper_bound = full_axis_len

        cropped_arr = np.zeros(shape=(1, 2 * half_span, 3))
        cropped_arr[0][0 + lower_padding:2 * half_span - upper_padding][:] = self.samples[0][lower_bound:upper_bound][:]
        self.samples = cropped_arr

    def scale_values(self, factor: int) -> None:
        self.samples = self.samples * factor

    # def filter_high_frequencies(self) -> None:
    #     ones_len = int(len(self.x) / 10)
    #     self.x = np.convolve(self.x, np.ones(ones_len), 'same') / ones_len
    #     self.y = np.convolve(self.y, np.ones(ones_len), 'same') / ones_len
    #     self.z = np.convolve(self.z, np.ones(ones_len), 'same') / ones_len

    def get_array(self) -> NDArray:
        return np.array(self.samples)

    def save_as_bitmap(self, savefile: str) -> None:
        im_array = np.array(self.samples)

        im_min = min(im_array)
        im_max = max(im_array)

        scaled_arr = ((im_array - im_min) / (im_max - im_min)) * 255
        scaled_arr = np.array(scaled_arr, dtype=np.uint8)

        im = Image.fromarray(scaled_arr, mode="RGB")
        im.save(savefile, format="BMP")


def plot_motion(motion: Motion, savefile: str):
    fig, ax = plt.subplots()
    axis_names = ["x", "y", "z"]
    x = motion.get_x()
    y = motion.get_y()
    z = motion.get_z()
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
    data.crop(data.get_global_extremum_position(), 40)
    plot_motion(data, "test_plot.jpg")
