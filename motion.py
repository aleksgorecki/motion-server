from __future__ import annotations
import json
from typing import Tuple
import PIL
import numpy as np
from PIL import Image
from numpy.typing import NDArray, ArrayLike
from matplotlib import pyplot as plt


class Motion:
    samples: NDArray  # samples[1][len_of_motion][3]

    def __init__(self, three_channel_array: NDArray) -> None:
        self.samples = three_channel_array

    @staticmethod
    def from_json(json_object) -> Motion:
        x = json_object['x']
        y = json_object['y']
        z = json_object['z']
        three_channel_array = np.empty(shape=(1, len(x), 3))
        for sample in range(len(x)):
            three_channel_array[0][sample][0] = x[sample]
            three_channel_array[0][sample][1] = y[sample]
            three_channel_array[0][sample][2] = z[sample]
        return Motion(
            three_channel_array
        )

    @staticmethod
    def from_separate_axes(x: NDArray, y: NDArray, z: NDArray):
        three_channel_array = np.empty(shape=(1, len(x), 3))
        for sample in range(len(x)):
            three_channel_array[0][sample][0] = x[sample]
            three_channel_array[0][sample][1] = y[sample]
            three_channel_array[0][sample][2] = z[sample]
        return Motion(three_channel_array)

    @staticmethod
    def from_bitmap(im: PIL.Image.Image):
        arr = np.asarray(im, dtype=float)
        return Motion(arr)

    def get_copy(self) -> Motion:
        return Motion(np.array(self.samples))

    def __len__(self):
        return self.samples.shape[1]

    def get_x(self) -> NDArray:
        return self.samples[0][:,0]

    def get_y(self) -> NDArray:
        return self.samples[0][:,1]

    def get_z(self) -> NDArray:
        return self.samples[0][:,2]

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

    def scale_values(self, factor: float) -> None:
        self.samples = self.samples * factor

    def horizontal_shift(self, ratio: float) -> None:
        new_arr = np.zeros(shape=self.samples.shape)
        threshold_idx = abs(int(ratio * len(self)))
        if ratio < 0:
            new_arr[0][0:len(self) - threshold_idx] = self.samples[0][threshold_idx:]
        else:
            new_arr[0][threshold_idx:] = self.samples[0][0:len(self) - threshold_idx]
        self.samples = new_arr

    def low_pass_filter(self, kernel_size: int = 5) -> None:
        axis_columns = (self.get_x(), self.get_y(), self.get_z())
        kernel = np.ones(kernel_size) / kernel_size
        filtered_columns = [np.convolve(column, kernel, mode="same") for column in axis_columns]
        self.samples = np.array([np.column_stack(filtered_columns)])

    def get_array(self) -> NDArray:
        return np.array(self.samples)

    def save_as_json(self, savefile: str) -> None:
        json_object = dict({
            "x": self.get_x().tolist(),
            "y": self.get_y().tolist(),
            "z": self.get_z().tolist()
        })
        with open(savefile, "w") as of:
            json.dump(json_object, of, indent=4)

    def save_as_bitmap(self, savefile: str) -> None:
        im_array = np.array(self.samples)

        im_min = im_array.min()
        im_max = im_array.max()

        scaled_arr = ((im_array - im_min) / (im_max - im_min)) * 255
        scaled_arr = np.array(scaled_arr, dtype=np.uint8)

        im = Image.fromarray(scaled_arr, mode="RGB")
        im.save(savefile, format="BMP")

    def normalize_separate(self) -> None:
        normalized_axes = []
        for axis in (self.get_x(), self.get_y(), self.get_z()):
            normalized_axes.append( (axis - axis.min() / axis.max() - axis.min()) )
        self.samples = Motion.from_separate_axes(normalized_axes[0], normalized_axes[1], normalized_axes[2]).samples

    def normalize_global(self) -> None:
        self.samples = ((self.samples - self.samples.min()) / (self.samples.max() - self.samples.min()))

    def save_as_plot(self, savefile: str):
        fig, ax = plt.subplots()
        axis_names = ["x", "y", "z"]
        for idx, motion_axis in enumerate((self.get_x(), self.get_y(), self.get_z())):
            ax.plot(range(len(self)), motion_axis, label=axis_names[idx])
        ax.legend()
        ax.set_ylabel("acceleration m/s^2")
        ax.set_xlabel("sample number")
        fig.savefig(savefile)
        plt.close(fig)

    def get_derivative(self) -> Motion:
        x, y, z = self.get_x(), self.get_y(), self.get_z()
        x = np.diff(x) / 1.0
        y = np.diff(y) / 1.0
        z = np.diff(z) / 1.0
        return Motion.from_separate_axes(x, y, z)

    def abs(self) -> None:
        self.samples = np.abs(self.samples)

    def get_integral(self) -> Motion:
        x, y, z = self.get_x(), self.get_y(), self.get_z()
        x = np.cumsum(x)
        y = np.cumsum(y)
        z = np.cumsum(z)
        return Motion.from_separate_axes(x, y, z)

    def get_standard_deviation(self) -> Tuple[float, float, float]:
        x, y, z = self.get_x(), self.get_y(), self.get_z()
        x = np.std(x)
        y = np.std(y)
        z = np.std(z)
        return float(x), float(y), float(z)

    def get_standard_deviation_global(self) -> float:
        std = np.std(self.samples)
        return float(std)


if __name__ == "__main__":
    with open("./basic2fixed/x_positive/x_positive_20221219_152944.json", "r") as f:
        m = Motion.from_json(json.load(f))
        m.save_as_plot("test_org.jpg")

        d = m.get_derivative()
        d.save_as_plot("test_der.jpg")
