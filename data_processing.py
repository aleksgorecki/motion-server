from __future__ import annotations
import numpy as np
from PIL import Image
import os
from matplotlib import pyplot as plt
import json
import io
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
        
        #cropped_axis = np.zeros(shape=(2 * half_span))

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


    def scale_down_values(self, factor: int):
        self.x = self.x / factor
        self.y = self.y / factor
        self.z = self.z / factor

    def filter_high_frequencies(self):
        ones_len = int(len(self.x) / 10)
        self.x = np.convolve(self.x, np.ones(ones_len), 'same') / ones_len
        self.y = np.convolve(self.y, np.ones(ones_len), 'same') / ones_len
        self.z = np.convolve(self.z, np.ones(ones_len), 'same') / ones_len

    def save_to_bitmap(self, savefile: str):
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


# def find_global_extremum_sample(x, y, z):

#     x_ex = (min(x), max(x))
#     y_ex = (min(y), max(y))
#     z_ex = (min(z), max(z))



#     dominant_axis = np.array()

#     arg_minimum = np.argmin(arr)
#     arg_maximum = np.argmax(arr)

#     if (arr[arg_maximum] > abs(arr[arg_minimum])):
#         return arg_maximum
#     else:
#         return arg_minimum


# def extract_extremum(arr: ArrayLike, neighborhood_half_span: int = 50, extremum_sample: int):
#     arg_minimum = np.argmin(arr)
#     arg_maximum = np.argmax(arr)

#     arg_extremum = None
#     extremum_neighborhood = np.zeros(shape=(neighborhood_half_span * 2))

#     if (arr[arg_maximum] > abs(arr[arg_minimum])):
#         arg_extremum = arg_maximum
#     else:
#         arg_extremum = arg_minimum
    
#     org_arr_low_bound = arg_extremum - neighborhood_half_span
#     org_arr_high_bound = arg_extremum + neighborhood_half_span

#     ext_arr_start = 0
#     ext_arr_end = len(extremum_neighborhood)

#     if org_arr_low_bound < 0:
#         ext_arr_start = abs(org_arr_low_bound)
#         org_arr_low_bound = 0
        
#     if org_arr_high_bound > len(arr):
#         ext_arr_end = len(extremum_neighborhood) - (org_arr_high_bound - len(arr))
#         org_arr_high_bound = len(arr)

#     extremum_neighborhood[ext_arr_start:ext_arr_end] = arr[org_arr_low_bound:org_arr_high_bound]

#     return extremum_neighborhood


# def motion_json_to_object(json_path):
#     with open(json_path, "r") as json_file:
#         return json.load(json_file)

# def json_data_to_array(json_data):
#     x = np.array(json_data["x"])
#     y = np.array(json_data["y"])
#     z = np.array(json_data["z"])

#     ret = np.zeros(shape=(1, 150, 3), dtype=float)
#     for sample_no in range(0, len(x) if len(x) <= 150 else 150):
#         ret[0][sample_no][0] = x[sample_no]
#         ret[0][sample_no][1] = y[sample_no]
#         ret[0][sample_no][2] = z[sample_no]

#     return ret

# def normalize_amplitude_values(arr):
#     global_min = arr.min()
#     global_max = arr.max()
#     return (((arr - global_min) / (global_max - global_min)))

# # def filter_high_frequency_oscillations(arr, ones_len):
# #     ret_arr = np.empty(dtype=float, shape=arr.shape)
# #     for row_idx, row in enumerate(arr):
# #         filtered_row = np.convolve(row, np.ones(ones_len), 'same') / ones_len
# #         ret_arr[row_idx] = np.array(filtered_row)
# #     return ret_arr

# def save_array_to_bitmap(arr, savefile):

#     scaled_arr = np.array(arr * 255, dtype=np.uint8)
#     im = Image.fromarray(scaled_arr, mode="RGB")
#     im.save(savefile, format="BMP")

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